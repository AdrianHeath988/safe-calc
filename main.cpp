#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <gmpxx.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <complex>
#include <unordered_map>
#include <mutex>
const double PI = 3.14159265358979323846;
const int N = 1 << 15; // 1<<15 is 32,768 
const double DELTA = 1152921504606584833.0; //2^60

// 60-bit primes (congruent to 1 mod 2N for future NTT compatibility)
const std::vector<uint64_t> Q_LIMBS = {
    1152921504606584833ULL, 1152921504598720513ULL, 1152921504597016577ULL,
    1152921504595968001ULL, 1152921504592822273ULL, 1152921504592429057ULL,
    1152921504589938689ULL, 1152921504586530817ULL, 1152921504583647233ULL,
    1152921504581419009ULL, 1152921504552910849ULL, 1152921504552583169ULL
};
// const std::vector<uint64_t> P_LIMBS = {
//     1152921504580894721ULL, 1152921504578666497ULL, 1152921504578273281ULL,
//     1152921504577748993ULL, 1152921504577486849ULL, 1152921504570802177ULL,
//     1152921504570277889ULL, 1152921504568836097ULL, 1152921504568442881ULL,
//     1152921504565559297ULL, 1152921504565166081ULL
// };
const std::vector<uint64_t> P_LIMBS = {
    1152921504580894721ULL // Just 1
};
struct Plaintext {
    std::vector<std::vector<uint64_t>> data;
    double scale = DELTA;
    Plaintext() : data(Q_LIMBS.size(), std::vector<uint64_t>(N, 0)){}; // init blank pt to have Q_LIMBS moduli, each modulus having all 0s
};
struct Ciphertext {
    std::vector<Plaintext> parts;
    double scale = DELTA;
    bool ntt_form = false;
};

inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)(((__uint128_t)a * b) % q);
}
// |d| <= q_k/2 < mod for all our ~2^60 primes, so no division needed
inline uint64_t lift_signed(int64_t d, uint64_t mod) {
    return d < 0 ? (uint64_t)((int64_t)mod + d) : (uint64_t)d;
}
struct RelinKeyP {
    // [decomposition_index_k][target_limb_i][N_polynomial]
    std::vector<std::vector<std::vector<uint64_t>>> b_Q, a_Q; 
    std::vector<std::vector<std::vector<uint64_t>>> b_P, a_P; 
};
std::vector<uint64_t> poly_mult_naive (const std::vector<uint64_t>& A, const std::vector<uint64_t>& B, uint64_t q){
    std::vector<__uint128_t> intermediate(2 * N -1, 0);
    #pragma omp parallel for
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            uint64_t product = (uint64_t)(((__uint128_t)A[i] * B[j]) % q);
            intermediate[i + j] = (intermediate[i+j]+product) % q;
        }
    }
    std::vector<uint64_t> out(N, 0);
    #pragma omp parallel for
    for(int i = 0; i < N; ++i) {
        uint64_t val = intermediate[i];
        if (i + N < intermediate.size()) {
            uint64_t upper = intermediate[i + N];
            //Why +q? Because we are usin uint64_t
            val = (val + q - upper) % q; // Apply X^N = -1
        }
        out[i] = val;
    }
    return out;
}


// helpers
inline uint64_t mod_exp(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (uint64_t)(((__uint128_t)res * base) % mod);
        base = (uint64_t)(((__uint128_t)base * base) % mod);
        exp /= 2;
    }
    return res;
}

inline uint64_t mod_inverse(uint64_t n, uint64_t mod) {
    return mod_exp(n, mod - 2, mod);
}

inline uint32_t bit_reverse(uint32_t val, int bit_length) {
    uint32_t res = 0;
    for (int i = 0; i < bit_length; i++) {
        res = (res << 1) | (val & 1);
        val >>= 1;
    }
    return res;
}

//Twiddle factor precomputation
struct NTTTables {
    std::vector<uint64_t> forward_twiddles, forward_twiddles_shoup;
    std::vector<uint64_t> inverse_twiddles, inverse_twiddles_shoup;
    uint64_t n_inv, n_inv_shoup;
};

inline uint64_t shoup_precomp(uint64_t W, uint64_t q) {
    return (uint64_t)(((__uint128_t)W << 64) / q);   // one-time cost, % is fine here
}
NTTTables precompute_ntt(uint64_t q, int N) {
    NTTTables tables;
    tables.forward_twiddles.resize(N);
    tables.inverse_twiddles.resize(N);
    tables.n_inv = mod_inverse(N, q);

    //primitive 2N-th root of unity modulo q
    uint64_t psi = 0;
    uint64_t test_base = 2;
    while (true) {
        psi = mod_exp(test_base, (q - 1) / (2 * N), q);
        if (mod_exp(psi, N, q) == q - 1) { // condition for psi^N == -1 mod q
            break;
        }
        test_base++;
    }

    uint64_t psi_inv = mod_inverse(psi, q);
    
    int logN = 0;
    while ((1ULL << logN) < N) logN++;

    //powers in bit-reversed order for linear memory access in loops
    //WE LOVE THE CACHE BE NICE TO THE CACHE
        for (int i = 0; i < N; i++) {
        uint32_t rev = bit_reverse(i, logN);
        tables.forward_twiddles[i] = mod_exp(psi, rev, q);
        tables.inverse_twiddles[i] = mod_exp(psi_inv, rev, q);
    }

    tables.forward_twiddles_shoup.resize(N);
    tables.inverse_twiddles_shoup.resize(N);
    for (int i = 0; i < N; i++) {
        tables.forward_twiddles_shoup[i] = shoup_precomp(tables.forward_twiddles[i], q);
        tables.inverse_twiddles_shoup[i] = shoup_precomp(tables.inverse_twiddles[i], q);
    }
    tables.n_inv_shoup = shoup_precomp(tables.n_inv, q);
    return tables;
}
const NTTTables& get_tables(uint64_t q, int N) {
    static std::unordered_map<uint64_t, NTTTables> cache;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    auto it = cache.find(q);
    if (it == cache.end()) it = cache.emplace(q, precompute_ntt(q, N)).first;
    return it->second; // references into unordered_map stay valid across rehash
}
inline uint64_t mul_shoup_lazy(uint64_t a, uint64_t W, uint64_t Wp, uint64_t q) {
    uint64_t hi = (uint64_t)(((__uint128_t)a * Wp) >> 64);
    return a * W - hi * q;                        // [0, 2q)
}
struct Red { uint64_t q, r64, r64p, op; };      // r64 = 2^64 mod q
inline Red make_red(uint64_t q) {
    uint64_t r = (uint64_t)((((__uint128_t)1) << 64) % q);
    return { q, r, shoup_precomp(r, q), shoup_precomp(1, q) };
}
// any 128-bit x -> [0,q). Two Shoup mults, no division.
inline uint64_t red128(__uint128_t x, const Red& c) {
    uint64_t a = mul_shoup_lazy((uint64_t)(x >> 64), c.r64, c.r64p, c.q);
    uint64_t b = mul_shoup_lazy((uint64_t)x, 1, c.op, c.q);
    uint64_t r = a + b;
    if (r >= 2 * c.q) r -= 2 * c.q;
    if (r >= c.q) r -= c.q;
    return r;
}
inline uint64_t addmod(uint64_t a, uint64_t b, uint64_t q) { uint64_t r = a + b; return r >= q ? r - q : r; }
inline uint64_t submod(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : a + q - b; }
inline uint64_t red_once(uint64_t x, uint64_t m) { uint64_t y = x - m; return y < x ? y : x; }
inline uint64_t mulc(uint64_t a, uint64_t c, uint64_t cp, uint64_t q) {   // a * constant
    uint64_t r = mul_shoup_lazy(a, c, cp, q); return r >= q ? r - q : r;
}
// ---------- AVX-512 helpers ----------
static inline __m512i mulhi64_v(__m512i a, __m512i b) {
    const __m512i m32 = _mm512_set1_epi64(0xFFFFFFFFULL);
    __m512i ah = _mm512_srli_epi64(a, 32), bh = _mm512_srli_epi64(b, 32);
    __m512i p0 = _mm512_mul_epu32(a, b);
    __m512i p1 = _mm512_mul_epu32(ah, b);
    __m512i p2 = _mm512_mul_epu32(a, bh);
    __m512i p3 = _mm512_mul_epu32(ah, bh);
    __m512i mid = _mm512_add_epi64(_mm512_srli_epi64(p0, 32),
                  _mm512_add_epi64(_mm512_and_si512(p1, m32), _mm512_and_si512(p2, m32)));
    return _mm512_add_epi64(p3,
           _mm512_add_epi64(_mm512_srli_epi64(p1, 32),
           _mm512_add_epi64(_mm512_srli_epi64(p2, 32), _mm512_srli_epi64(mid, 32))));
}
// a*W - floor(a*Wp/2^64)*q  (mod 2^64)  ->  [0, 2q)
static inline __m512i mulshoup_v(__m512i a, __m512i W, __m512i Wp, __m512i q) {
    __m512i hi = mulhi64_v(a, Wp);
    return _mm512_sub_epi64(_mm512_mullo_epi64(a, W), _mm512_mullo_epi64(hi, q));
}
static inline __m512i red_once_v(__m512i x, __m512i m) {   // x>=m ? x-m : x
    return _mm512_min_epu64(x, _mm512_sub_epi64(x, m));
}
static inline void fwd_bfly_v(__m512i& X, __m512i& Y, __m512i W, __m512i Wp, __m512i q, __m512i q2) {
    X = red_once_v(X, q2);
    __m512i Tm = mulshoup_v(Y, W, Wp, q);
    Y = _mm512_add_epi64(_mm512_sub_epi64(X, Tm), q2);
    X = _mm512_add_epi64(X, Tm);
}
static inline void inv_bfly_v(__m512i& X, __m512i& Y, __m512i W, __m512i Wp, __m512i q, __m512i q2) {
    __m512i S = _mm512_add_epi64(X, Y);
    __m512i D = _mm512_add_epi64(_mm512_sub_epi64(X, Y), q2);
    X = red_once_v(S, q2);
    Y = mulshoup_v(D, W, Wp, q);
}
// 4 consecutive twiddles w0..w3 -> [w0 w0 w1 w1 w2 w2 w3 w3]
static inline __m512i dup4_v(const uint64_t* p) {
    __m512i w = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)p));
    return _mm512_permutexvar_epi64(_mm512_setr_epi64(0,0,1,1,2,2,3,3), w);
}
// 2 twiddles -> [a a a a b b b b]
static inline __m512i split2_v(uint64_t a, uint64_t b) {
    return _mm512_mask_set1_epi64(_mm512_set1_epi64((long long)a), 0xF0, (long long)b);
}

// ---------- forward NTT (Cooley-Tukey), AVX-512 ----------
//I LOVE __restrict I LOVE __restrict I LOVE __restrict
void ntt(std::vector<uint64_t>& a, const NTTTables& T, uint64_t q, int N) {
    const __m512i vq = _mm512_set1_epi64((long long)q), vq2 = _mm512_set1_epi64((long long)(2 * q));
    uint64_t* __restrict A = a.data();
    const uint64_t* __restrict Wt = T.forward_twiddles.data();
    const uint64_t* __restrict Wps = T.forward_twiddles_shoup.data();

    // stages with t >= 8
    int t = N, m = 1;
    for (; m <= N / 16; m <<= 1) {
        t >>= 1;
        for (int i = 0; i < m; i++) {
            const __m512i W = _mm512_set1_epi64((long long)Wt[m + i]);
            const __m512i Wp = _mm512_set1_epi64((long long)Wps[m + i]);
            uint64_t* x = A + 2 * (size_t)i * t;
            uint64_t* y = x + t;
            //UNROLL THIS LOOP OR I WILL NEVER GIVE YOU A #likely AGAIN
            #pragma unroll 4
            for (int j = 0; j < t; j += 8) {
                __m512i X = _mm512_loadu_si512(x + j), Y = _mm512_loadu_si512(y + j);
                fwd_bfly_v(X, Y, W, Wp, vq, vq2);
                _mm512_storeu_si512(x + j, X);
                _mm512_storeu_si512(y + j, Y);
            }
        }
    }

    // last three stages (t=4,2,1) in registers, 16 coefficients per iteration
    const __m512i iX1 = _mm512_setr_epi64(0,1,2,3,8,9,10,11),  iY1 = _mm512_setr_epi64(4,5,6,7,12,13,14,15);
    const __m512i iX2 = _mm512_setr_epi64(0,1,8,9,4,5,12,13),  iY2 = _mm512_setr_epi64(2,3,10,11,6,7,14,15);
    const __m512i iX3 = _mm512_setr_epi64(0,8,2,10,4,12,6,14), iY3 = _mm512_setr_epi64(1,9,3,11,5,13,7,15);
    const __m512i iO0 = _mm512_setr_epi64(0,8,1,9,2,10,3,11),  iO1 = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
    const int c4 = N / 8, c2 = N / 4, c1 = N / 2;

    for (int c = 0; c < N / 16; c++) {
        uint64_t* p = A + 16 * (size_t)c;
        __m512i v0 = _mm512_loadu_si512(p), v1 = _mm512_loadu_si512(p + 8);

        // t = 4 : twiddle index c4 + 2c (+1 for the second 8-block)
        __m512i X = _mm512_permutex2var_epi64(v0, iX1, v1), Y = _mm512_permutex2var_epi64(v0, iY1, v1);
        fwd_bfly_v(X, Y, split2_v(Wt[c4 + 2*c], Wt[c4 + 2*c + 1]),
                         split2_v(Wps[c4 + 2*c], Wps[c4 + 2*c + 1]), vq, vq2);

        // t = 2 : twiddle index c2 + 4c .. +3, each used twice
        __m512i X2 = _mm512_permutex2var_epi64(X, iX2, Y), Y2 = _mm512_permutex2var_epi64(X, iY2, Y);
        fwd_bfly_v(X2, Y2, dup4_v(Wt + c2 + 4*c), dup4_v(Wps + c2 + 4*c), vq, vq2);

        // t = 1 : twiddle index c1 + 8c .. +7
        __m512i X3 = _mm512_permutex2var_epi64(X2, iX3, Y2), Y3 = _mm512_permutex2var_epi64(X2, iY3, Y2);
        fwd_bfly_v(X3, Y3, _mm512_loadu_si512(Wt + c1 + 8*c), _mm512_loadu_si512(Wps + c1 + 8*c), vq, vq2);

        // final reduction to [0, q)
        X3 = red_once_v(red_once_v(X3, vq2), vq);
        Y3 = red_once_v(red_once_v(Y3, vq2), vq);

        _mm512_storeu_si512(p,     _mm512_permutex2var_epi64(X3, iO0, Y3));
        _mm512_storeu_si512(p + 8, _mm512_permutex2var_epi64(X3, iO1, Y3));
    }
}

// ---------- inverse NTT (Gentleman-Sande), AVX-512 ----------
void intt(std::vector<uint64_t>& a, const NTTTables& T, uint64_t q, int N) {
    const __m512i vq = _mm512_set1_epi64((long long)q), vq2 = _mm512_set1_epi64((long long)(2 * q));
    uint64_t* __restrict A = a.data();
    const uint64_t* __restrict Wt = T.inverse_twiddles.data();
    const uint64_t* __restrict Wps = T.inverse_twiddles_shoup.data();

    // first three stages (t=1,2,4) in registers
    const __m512i iX1 = _mm512_setr_epi64(0,2,4,6,8,10,12,14), iY1 = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
    const __m512i iX2 = _mm512_setr_epi64(0,8,2,10,4,12,6,14), iY2 = _mm512_setr_epi64(1,9,3,11,5,13,7,15);
    const __m512i iX3 = _mm512_setr_epi64(0,1,8,9,4,5,12,13),  iY3 = _mm512_setr_epi64(2,3,10,11,6,7,14,15);
    const __m512i iO0 = _mm512_setr_epi64(0,1,2,3,8,9,10,11),  iO1 = _mm512_setr_epi64(4,5,6,7,12,13,14,15);
    const int h1 = N / 2, h2 = N / 4, h4 = N / 8;

    for (int c = 0; c < N / 16; c++) {
        uint64_t* p = A + 16 * (size_t)c;
        __m512i v0 = _mm512_loadu_si512(p), v1 = _mm512_loadu_si512(p + 8);

        // t = 1
        __m512i X = _mm512_permutex2var_epi64(v0, iX1, v1), Y = _mm512_permutex2var_epi64(v0, iY1, v1);
        inv_bfly_v(X, Y, _mm512_loadu_si512(Wt + h1 + 8*c), _mm512_loadu_si512(Wps + h1 + 8*c), vq, vq2);

        // t = 2
        __m512i X2 = _mm512_permutex2var_epi64(X, iX2, Y), Y2 = _mm512_permutex2var_epi64(X, iY2, Y);
        inv_bfly_v(X2, Y2, dup4_v(Wt + h2 + 4*c), dup4_v(Wps + h2 + 4*c), vq, vq2);

        // t = 4
        __m512i X3 = _mm512_permutex2var_epi64(X2, iX3, Y2), Y3 = _mm512_permutex2var_epi64(X2, iY3, Y2);
        inv_bfly_v(X3, Y3, split2_v(Wt[h4 + 2*c], Wt[h4 + 2*c + 1]),
                           split2_v(Wps[h4 + 2*c], Wps[h4 + 2*c + 1]), vq, vq2);

        _mm512_storeu_si512(p,     _mm512_permutex2var_epi64(X3, iO0, Y3));
        _mm512_storeu_si512(p + 8, _mm512_permutex2var_epi64(X3, iO1, Y3));
    }

    // remaining stages, t >= 8
    int t = 8;
    for (int m = N / 8; m >= 2; m >>= 1) {
        int h = m >> 1;
        for (int i = 0; i < h; i++) {
            const __m512i W = _mm512_set1_epi64((long long)Wt[h + i]);
            const __m512i Wp = _mm512_set1_epi64((long long)Wps[h + i]);
            uint64_t* x = A + 2 * (size_t)i * t;
            uint64_t* y = x + t;
            #pragma unroll 4
            for (int j = 0; j < t; j += 8) {
                __m512i X = _mm512_loadu_si512(x + j), Y = _mm512_loadu_si512(y + j);
                inv_bfly_v(X, Y, W, Wp, vq, vq2);
                _mm512_storeu_si512(x + j, X);
                _mm512_storeu_si512(y + j, Y);
            }
        }
        t <<= 1;
    }

    // normalize by n_inv, reduce to [0, q)
    const __m512i Wn = _mm512_set1_epi64((long long)T.n_inv), Wnp = _mm512_set1_epi64((long long)T.n_inv_shoup);
    for (int i = 0; i < N; i += 8) {
        __m512i x = mulshoup_v(_mm512_loadu_si512(A + i), Wn, Wnp, vq);
        _mm512_storeu_si512(A + i, red_once_v(x, vq));
    }
}

//Cooley-Tukey Forward NTT
void ntt_scalar(std::vector<uint64_t>& a, const NTTTables& T, uint64_t q, int N) {
    const uint64_t q2 = 2 * q;
    uint64_t* __restrict A = a.data();
    const uint64_t* __restrict Wt = T.forward_twiddles.data();
    const uint64_t* __restrict Wps = T.forward_twiddles_shoup.data();
    int t = N;
    for (int m = 1; m < N / 2; m <<= 1) {
        t >>= 1;
        for (int i = 0; i < m; i++) {
            const uint64_t W = Wt[m + i], Wp = Wps[m + i];
            uint64_t* __restrict x = A + 2 * i * t;
            uint64_t* __restrict y = x + t;
            for (int j = 0; j < t; j++) {
                uint64_t X = red_once(x[j], q2);
                uint64_t Tm = mul_shoup_lazy(y[j], W, Wp, q);
                x[j] = X + Tm;
                y[j] = X - Tm + q2;
            }
        }
    }
    for (int i = 0; i < N / 2; i++) {            // last stage + final reduction
        const uint64_t W = Wt[N / 2 + i], Wp = Wps[N / 2 + i];
        uint64_t X = red_once(A[2 * i], q2);
        uint64_t Tm = mul_shoup_lazy(A[2 * i + 1], W, Wp, q);
        uint64_t u = X + Tm, v = X - Tm + q2;
        A[2 * i]     = red_once(red_once(u, q2), q);
        A[2 * i + 1] = red_once(red_once(v, q2), q);
    }
}
//gentelemane snade intt
void intt_scalar(std::vector<uint64_t>& a, const NTTTables& T, uint64_t q, int N) {
    const uint64_t q2 = 2 * q;
    uint64_t* __restrict A = a.data();
    const uint64_t* __restrict Wt = T.inverse_twiddles.data();
    const uint64_t* __restrict Wps = T.inverse_twiddles_shoup.data();
    int t = 1;
    for (int m = N; m > 1; m >>= 1) {
        int h = m >> 1;
        for (int i = 0; i < h; i++) {
            const uint64_t W = Wt[h + i], Wp = Wps[h + i];
            uint64_t* __restrict x = A + 2 * i * t;
            uint64_t* __restrict y = x + t;
            for (int j = 0; j < t; j++) {
                uint64_t X = x[j], Y = y[j];
                x[j] = red_once(X + Y, q2);
                y[j] = mul_shoup_lazy(X - Y + q2, W, Wp, q);
            }
        }
        t <<= 1;
    }
    for (int i = 0; i < N; i++)
        A[i] = red_once(mul_shoup_lazy(A[i], T.n_inv, T.n_inv_shoup, q), q);
}

std::vector<uint64_t> ntt_mul(const std::vector<uint64_t>& A, const std::vector<uint64_t>& B, uint64_t q) {
    int N = A.size();
    
    //cache twiddles per prime to guarantee O(N log N) speed on repeat calls
    static std::unordered_map<uint64_t, NTTTables> cache;
    const NTTTables& tables = get_tables(q, N);

    //mutable copies
    std::vector<uint64_t> a_ntt = A;
    std::vector<uint64_t> b_ntt = B;
    
    ntt(a_ntt, tables, q, N);
    ntt(b_ntt, tables, q, N);
    
    std::vector<uint64_t> c_ntt(N);
    
    //Pointwise multiplication 
    //(A and B are in bit-reversed order, which is perfectly fine!)
    for(int i = 0; i < N; i++) {
        c_ntt[i] = (uint64_t)(((__uint128_t)a_ntt[i] * b_ntt[i]) % q);
    }
    
    intt(c_ntt, tables, q, N);
    
    return c_ntt;
}
inline uint64_t wrap(int64_t x, uint64_t q) {
    int64_t rem = x % (int64_t)q;
    return rem < 0 ? (uint64_t)(rem + q) : (uint64_t)rem;
}
double g_ph[6] = {0};
const char* g_phn[6] = {"tensor","digit+intt","key product","mod-down","combine","rescale"};
class CKKSEngine {
public:
    std::vector<const NTTTables*> tabQ, tabP;
    std::vector<Red> redQ; Red redP;
    std::vector<uint64_t> qhi_p;                          // shoup of q_hat_inv_mod_q[nQ-1][k]
    std::vector<std::vector<uint64_t>> Pinv_p, qinv_p;    // shoup of P_inv_mod_q[lvl][i], q_inv_mod_q[l][i]
    //setup phase for relin fast auxillirary bassi switch for later
    // --- Public Class Variables ---
    // [level][i]
    std::vector<std::vector<uint64_t>> q_hat_inv_mod_q;
    std::vector<std::vector<uint64_t>> P_mod_q, P_inv_mod_q;
    
    //[level][i][j]
    std::vector<std::vector<std::vector<uint64_t>>> q_hat_mod_p;
    //[level][j]
    std::vector<std::vector<uint64_t>> Q_mod_p;

    //static p tables(independent of Q limb drops)
    std::vector<uint64_t> p_hat_inv_mod_p;
    std::vector<std::vector<uint64_t>> p_hat_mod_q;
    
    //rescale dropping table
    std::vector<std::vector<uint64_t>> q_inv_mod_q; // [limb_to_drop][remaining_limb]


    //c_Q: ntt form, modified in place.  c_P: coefficient form.
    void mod_down_ntt(std::vector<std::vector<uint64_t>>& c_Q, const std::vector<uint64_t>& c_P,
                      const std::vector<std::vector<uint64_t>>& d) {
        size_t L = c_Q.size(), lvl = L - 1;
        uint64_t p0 = P_LIMBS[0], p_half = p0 / 2;
        #pragma omp parallel for
        for (int i = 0; i < (int)L; i++) {
            uint64_t qi = Q_LIMBS[i];
            std::vector<uint64_t> e(N);
            for (int j = 0; j < N; j++) {
                uint64_t ep = c_P[j];
                e[j] = (ep > p_half) ? qi - (p0 - ep) : ep;
            }
            ntt(e, *tabQ[i], qi, N);
            const uint64_t c = P_inv_mod_q[lvl][i], cp = Pinv_p[lvl][i];
            for (int j = 0; j < N; j++)
                c_Q[i][j] = addmod(mulc(submod(c_Q[i][j], e[j], qi), c, cp, qi), d[i][j], qi);
        }
    }
    void setup_rns_tables() {
        size_t nQ = Q_LIMBS.size(), nP = P_LIMBS.size();

        //resize tables
        q_hat_inv_mod_q.resize(nQ);
        q_hat_mod_p.resize(nQ);
        Q_mod_p.resize(nQ);
        P_mod_q.resize(nQ);
        P_inv_mod_q.resize(nQ);

        mpz_class P_val = 1; 
        for (auto p : P_LIMBS) P_val *= p;

        //q tables
        for (size_t lvl = 0; lvl < nQ; lvl++) {
            size_t current_limbs = lvl + 1;
            
            mpz_class Q_val = 1; 
            for (size_t i = 0; i < current_limbs; i++) Q_val *= Q_LIMBS[i];

            q_hat_inv_mod_q[lvl].resize(current_limbs);
            q_hat_mod_p[lvl].resize(current_limbs, std::vector<uint64_t>(nP));
            Q_mod_p[lvl].resize(nP);
            P_mod_q[lvl].resize(current_limbs);
            P_inv_mod_q[lvl].resize(current_limbs);
            
            for (size_t i = 0; i < current_limbs; i++) {
                mpz_class Qi = Q_val / Q_LIMBS[i];
                mpz_class inv; 
                mpz_invert(inv.get_mpz_t(), Qi.get_mpz_t(), mpz_class(Q_LIMBS[i]).get_mpz_t());
                q_hat_inv_mod_q[lvl][i] = inv.get_ui();
                
                for (size_t j = 0; j < nP; j++) {
                    q_hat_mod_p[lvl][i][j] = mpz_class(Qi % P_LIMBS[j]).get_ui();
                }
                
                P_mod_q[lvl][i] = mpz_class(P_val % Q_LIMBS[i]).get_ui();
                
                mpz_class P_inv;
                mpz_invert(P_inv.get_mpz_t(), P_val.get_mpz_t(), mpz_class(Q_LIMBS[i]).get_mpz_t());
                P_inv_mod_q[lvl][i] = P_inv.get_ui();
            }
            
            for (size_t j = 0; j < nP; j++) {
                Q_mod_p[lvl][j] = mpz_class(Q_val % P_LIMBS[j]).get_ui();
            }
        }

        //P tables
        p_hat_inv_mod_p.resize(nP); 
        p_hat_mod_q.resize(nP, std::vector<uint64_t>(nQ));
        
        for (size_t k = 0; k < nP; k++) {
            mpz_class Pk = P_val / P_LIMBS[k];
            mpz_class inv; 
            mpz_invert(inv.get_mpz_t(), Pk.get_mpz_t(), mpz_class(P_LIMBS[k]).get_mpz_t());
            p_hat_inv_mod_p[k] = inv.get_ui();
            
            for (size_t i = 0; i < nQ; i++) {
                p_hat_mod_q[k][i] = mpz_class(Pk % Q_LIMBS[i]).get_ui();
            }
        }

        //inverse of dropped limbs
        q_inv_mod_q.resize(nQ, std::vector<uint64_t>(nQ));
        for (size_t l = 1; l < nQ; l++) {
            for (size_t i = 0; i < l; i++) {
                mpz_class inv; 
                mpz_invert(inv.get_mpz_t(), mpz_class(Q_LIMBS[l]).get_mpz_t(), mpz_class(Q_LIMBS[i]).get_mpz_t());
                q_inv_mod_q[l][i] = inv.get_ui(); 
            }
        }
        tabQ.resize(nQ); tabP.resize(nP);
        for (size_t i = 0; i < nQ; i++) tabQ[i] = &get_tables(Q_LIMBS[i], N);
        for (size_t i = 0; i < nP; i++) tabP[i] = &get_tables(P_LIMBS[i], N);
        redQ.resize(nQ);
        for (size_t i = 0; i < nQ; i++) redQ[i] = make_red(Q_LIMBS[i]);
        redP = make_red(P_LIMBS[0]);
                qhi_p.resize(nQ);
        Pinv_p.assign(nQ, std::vector<uint64_t>(nQ, 0));
        qinv_p.assign(nQ, std::vector<uint64_t>(nQ, 0));
        for (size_t i = 0; i < nQ; i++) {
            qhi_p[i] = shoup_precomp(q_hat_inv_mod_q[nQ - 1][i], Q_LIMBS[i]);
            for (size_t lvl = i; lvl < nQ; lvl++)
                Pinv_p[lvl][i] = shoup_precomp(P_inv_mod_q[lvl][i], Q_LIMBS[i]);
            for (size_t l = i + 1; l < nQ; l++)
                qinv_p[l][i] = shoup_precomp(q_inv_mod_q[l][i], Q_LIMBS[i]);
        }
    }
    //key is P * q_k tilda *s^2 for every decomposed limb 
    //rewriting alot of enc algo (look at that first)
    //
    RelinKeyP generate_relin_key(const std::vector<int>& secret) {
        RelinKeyP rlk;
        size_t nQ = Q_LIMBS.size();
        size_t nP = P_LIMBS.size(); // This is now 1
        
        rlk.b_Q.resize(nQ, std::vector<std::vector<uint64_t>>(nQ, std::vector<uint64_t>(N)));
        rlk.a_Q.resize(nQ, std::vector<std::vector<uint64_t>>(nQ, std::vector<uint64_t>(N)));
        rlk.b_P.resize(nQ, std::vector<std::vector<uint64_t>>(nP, std::vector<uint64_t>(N)));
        rlk.a_P.resize(nQ, std::vector<std::vector<uint64_t>>(nP, std::vector<uint64_t>(N)));

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::normal_distribution<double> gaussian_dist(0.0, 1.78885);

        //s^2
        std::vector<int64_t> s2(N, 0);
        for(int i = 0; i < N; ++i) {
            for(int j = 0; j < N; ++j) {
                int64_t prod = secret[i] * secret[j];
                if (i + j < N) s2[i + j] += prod;
                else s2[(i + j) % N] -= prod; 
            }
        }

        // Top-level Q calculation
        mpz_class Q_val = 1; 
        for (auto q : Q_LIMBS) Q_val *= q;
        mpz_class P_val = P_LIMBS[0];

        // Generate a separate key for each digit k
        for (size_t k = 0; k < nQ; k++) {
            // Calculate P * (Q / q_k) mod q_k
            mpz_class Q_k = Q_val / Q_LIMBS[k];
            uint64_t PQ_k_mod_qk = mpz_class(P_val * Q_k % Q_LIMBS[k]).get_ui();

            std::vector<int64_t> global_e(N);
            for(int j = 0; j < N; j++) global_e[j] = std::round(gaussian_dist(gen));

            // Generate the Q-basis parts of the key
            for(size_t i = 0; i < nQ; i++) {
                uint64_t mod = Q_LIMBS[i];
                std::uniform_int_distribution<uint64_t> dist(0, mod - 1);
                std::vector<uint64_t> a(N), s_mod(N), s2_mod(N);
                
                for(int j = 0; j < N; j++) {
                    a[j] = dist(gen); 
                    s_mod[j] = wrap(secret[j], mod); 
                    s2_mod[j] = wrap(s2[j], mod);
                }
                std::vector<uint64_t> as = ntt_mul(a, s_mod, mod);
                
                std::vector<uint64_t> base_vec(N);
                for(int j = 0; j < N; j++) {
                    uint64_t base = (mod - as[j]) % mod;
                    base = (base + wrap(global_e[j], mod)) % mod;
                    
                    if (i == k) {
                        uint64_t ps2 = (uint64_t)(((__uint128_t)PQ_k_mod_qk * s2_mod[j]) % mod);
                        base = (base + ps2) % mod;
                    }
                    base_vec[j] = base;
                }
                
                // CONVERT TO NTT DOMAIN BEFORE SAVING
                const NTTTables& tables = get_tables(mod, N); // Use cache here in real code
                ntt(base_vec, tables, mod, N);
                ntt(a, tables, mod, N);
                
                for(int j = 0; j < N; j++) {
                    rlk.b_Q[k][i][j] = base_vec[j];
                    rlk.a_Q[k][i][j] = a[j];
                }
            }

            // Generate the P-basis parts (no secret injection needed here mathematically)
            for(size_t i = 0; i < nP; i++) {
                uint64_t mod = P_LIMBS[i];
                std::uniform_int_distribution<uint64_t> dist(0, mod - 1);
                std::vector<uint64_t> a(N), s_mod(N);
                
                for(int j = 0; j < N; j++) {
                    a[j] = dist(gen); 
                    s_mod[j] = wrap(secret[j], mod);
                }
                
                // ntt_mul returns the normal-format product a*s
                std::vector<uint64_t> as = ntt_mul(a, s_mod, mod);
                
                std::vector<uint64_t> base_vec(N);
                for(int j = 0; j < N; j++) {
                    uint64_t base = (mod - as[j]) % mod;
                    base_vec[j] = (base + wrap(global_e[j], mod)) % mod;
                }
                
                // CONVERT TO NTT DOMAIN BEFORE SAVING
                NTTTables tables = precompute_ntt(mod, N); 
                ntt(base_vec, tables, mod, N);
                ntt(a, tables, mod, N);
                
                for(int j = 0; j < N; j++) {
                    rlk.b_P[k][i][j] = base_vec[j];
                    rlk.a_P[k][i][j] = a[j];
                }
            }
        }
        return rlk;
    }
    //extend to QUP basis using precomputed swithing
    //do this so later we can use AVX simd
    std::vector<std::vector<uint64_t>> fast_mod_up(const std::vector<std::vector<uint64_t>>& d_Q) {
        size_t current_limbs = d_Q.size();
        size_t lvl = current_limbs - 1; 
        size_t nP = P_LIMBS.size();
        std::vector<std::vector<uint64_t>> d_P(nP, std::vector<uint64_t>(N, 0));
        #pragma omp parallel for
        for (int j = 0; j < N; j++) {
            uint64_t sum_frac = 0, v = 0;
            std::vector<uint64_t> x_prime(current_limbs);
            
            for (size_t i = 0; i < current_limbs; i++) {
                uint64_t qi = Q_LIMBS[i];
                x_prime[i] = (uint64_t)(((__uint128_t)d_Q[i][j] * q_hat_inv_mod_q[lvl][i]) % qi);
                uint64_t frac = (uint64_t)(((__uint128_t)x_prime[i] << 64) / qi);
                uint64_t old = sum_frac;
                sum_frac += frac;
                if (sum_frac < old) v++;
            }
            uint64_t old = sum_frac;
            sum_frac += (1ULL << 63); 
            if (sum_frac < old) v++;

            for (size_t k = 0; k < nP; k++) {
                uint64_t pk = P_LIMBS[k];
                __uint128_t sum = 0;
                for (size_t i = 0; i < current_limbs; i++) {
                    sum += (__uint128_t)x_prime[i] * q_hat_mod_p[lvl][i][k]; 
                }
                uint64_t res = sum % pk;
                uint64_t sub = ((__uint128_t)v * Q_mod_p[lvl][k]) % pk;
                d_P[k][j] = (res + pk - sub) % pk;
            }
        }
        return d_P;
    }

    std::vector<std::vector<uint64_t>> fast_mod_down(
        const std::vector<std::vector<uint64_t>>& c_Q, const std::vector<uint64_t>& c_P) {
        
        size_t current_limbs = c_Q.size();
        size_t lvl = current_limbs - 1;
        std::vector<std::vector<uint64_t>> res_Q(current_limbs, std::vector<uint64_t>(N, 0));
        
        uint64_t p_0 = P_LIMBS[0];
        uint64_t p_half = p_0 / 2;

        #pragma omp parallel for
        for (int j = 0; j < N; j++) {
            uint64_t e_p = c_P[j]; // c_P is now correctly accessed as 1D
            
            for (size_t i = 0; i < current_limbs; i++) {
                uint64_t qi = Q_LIMBS[i];
                uint64_t e_qi;
                
                // Center the noise digit around 0 before reducing it mod q_i
                if (e_p > p_half) {
                    e_qi = (qi - ((p_0 - e_p) % qi)) % qi;
                } else {
                    e_qi = e_p % qi;
                }
                
                uint64_t diff = (c_Q[i][j] + qi - e_qi) % qi;
                res_Q[i][j] = (uint64_t)(((__uint128_t)diff * P_inv_mod_q[lvl][i]) % qi);
            }
        }
        return res_Q;
    }
    Plaintext encode(std::vector<uint64_t>& vals){
        std::vector<std::complex<double>> y(N);   
        for (int j = 0; j < N / 2; ++j) {
            y[j] = std::complex<double>(vals[j], 0.0);
            y[j + N / 2] = std::conj(y[j]); 
        }

        Plaintext pt;  
        
        //compute the roots dynamically, no w_matrix (it was eating ram like chrome)
        //just do it each time like a normal person
        #pragma omp parallel for
        for(int i = 0; i < N; i++){
            std::complex<double> sum(0.0, 0.0);
            for (int j = 0; j < N; ++j) {
                double angle;
                if (j < N / 2) {
                    //angle conj(roots[j])^i
                    angle = -i * (2.0 * j + 1.0) * PI / N;
                } else {
                    //now for roots[j-N/2]^i
                    angle = i * (2.0 * (j - N / 2) + 1.0) * PI / N;
                }
                
                std::complex<double> w_ij(cos(angle), sin(angle));
                sum += w_ij * y[j];
            }
            
            int64_t mi = std::round((sum.real() / N) * DELTA);
            for (size_t k = 0; k < Q_LIMBS.size(); ++k) {
                pt.data[k][i] = wrap(mi, Q_LIMBS[k]);
            }
        }
        return pt;
    }
    std::vector<double> decode(const Plaintext& pt) {
        //crt combine
        size_t num_limbs = pt.data.size();
        
        //gmp bigint
        mpz_class Q = 1;
        for(size_t i = 0; i < num_limbs; ++i) {
            Q *= Q_LIMBS[i];
        }
        //shift lower half into negatives
        mpz_class Q_half = Q / 2;

        std::vector<mpz_class> m_big(N, 0);
        
        //recontrsuct
        for(size_t i = 0; i < num_limbs; ++i) {
            mpz_class qi = Q_LIMBS[i];
            mpz_class Mi = Q / qi;
            
            //yi = Mi^(-1) mod qi
            mpz_class yi;
            mpz_invert(yi.get_mpz_t(), Mi.get_mpz_t(), qi.get_mpz_t());
            
            for(int j = 0; j < N; ++j) {
                mpz_class term = pt.data[i][j];
                term = (term * Mi * yi) % Q;
                m_big[j] = (m_big[j] + term) % Q;
            }
        }

        //Undo the wrap
        std::vector<double> m(N);
        for(int j = 0; j < N; ++j) {
            if (m_big[j] > Q_half) {
                m_big[j] -= Q;
            }
            //un-rns coefficient is now small enough
            m[j] = m_big[j].get_d();
        }

        //evaluating the polynomial at the roots (mult by W* squiggle)
        std::vector<double> res(N / 2);
        std::complex<double> zeta(cos(PI / N), sin(PI / N));
        
        //only evaluate at the first N/2 roots because those hold our original real values
        for (int j = 0; j < N / 2; ++j) {
            std::complex<double> root = std::pow(zeta, 2 * j + 1);
            std::complex<double> evaluated(0.0, 0.0);
            
            // sum( m[k] * root^k )
            for (int k = 0; k < N; ++k) {
                std::complex<double> root_k = std::pow(root, k);
                evaluated += m[k] * root_k; 
            }
            //dynamic scale
            res[j] = evaluated.real() / pt.scale;
        }
        
        return res;
    }
    Ciphertext encrypt(const Plaintext& pt, const std::vector<int>& secret){
        Ciphertext ct;
        ct.parts.resize(2, Plaintext());
        //setup RNG for sampling the masking polynomial a
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::vector<int64_t> global_e(N);
        std::normal_distribution<double> gaussian_dist(0.0, 1.78885);
        for (int j = 0; j < N; ++j) {
            global_e[j] = std::round(gaussian_dist(gen));
        }
        for(int i=0;i<Q_LIMBS.size();i++){
            uint64_t qi = Q_LIMBS[i];
            //gen a
            std::uniform_int_distribution<uint64_t> distrib(0, qi - 1);
            
            std::vector<uint64_t> a(N);
            std::vector<uint64_t> e_limb(N);
            for (int j = 0; j < N; ++j) {
                a[j] = distrib(gen);
                
                e_limb[j] = wrap(global_e[j], qi);
            }
            std::vector<uint64_t> sk_limb(N);
            for (int j = 0; j < N; ++j) {
                //warp sk so no neg
                sk_limb[j] = wrap(secret[j], qi); 
            }

            //A * S
            std::vector<uint64_t> as = ntt_mul(a, sk_limb, qi);

            //c0 and c1
            for (int j = 0; j < N; ++j) {
                //c0 = m - a*s + e mod qi
                uint64_t base_c0 = (pt.data[i][j] + qi - as[j]) % qi;
                ct.parts[0].data[i][j] = (base_c0 + e_limb[j]) % qi;
                
                // c1 = a
                ct.parts[1].data[i][j] = a[j];
            }
            const NTTTables& T = get_tables(qi, N);
            ntt(ct.parts[0].data[i], T, qi, N);
            ntt(ct.parts[1].data[i], T, qi, N);
        }
        ct.scale = pt.scale;
        ct.ntt_form = true;
        return ct;
    }
    Plaintext decrypt(const Ciphertext& ct, const std::vector<int>& secret) {
        Plaintext pt;
        //replace so it mults in ntt form and intts once
        size_t L = ct.parts[0].data.size();
        pt.data.resize(L, std::vector<uint64_t>(N, 0));
        for (size_t i = 0; i < L; ++i) {
            uint64_t qi = Q_LIMBS[i];
            const NTTTables& T = *tabQ[i];
            std::vector<uint64_t> sk(N);
            for (int j = 0; j < N; ++j) sk[j] = wrap(secret[j], qi);
            ntt(sk, T, qi, N);
            for (int j = 0; j < N; ++j)
                pt.data[i][j] = (ct.parts[0].data[i][j] + mulmod(ct.parts[1].data[i][j], sk[j], qi)) % qi;
            intt(pt.data[i], T, qi, N);
        }
        pt.scale = ct.scale;
        return pt;
    }
    Ciphertext add(const Ciphertext& c1, const Ciphertext& c2){
        Ciphertext ct;

        ct.parts.resize(2, Plaintext());
        size_t current_limbs = std::min(c1.parts[0].data.size(), c2.parts[0].data.size());
        ct.parts[0].data.resize(current_limbs, std::vector<uint64_t>(N, 0));
        ct.parts[1].data.resize(current_limbs, std::vector<uint64_t>(N, 0));
        for(int i=0;i<current_limbs;i++){    //per limb
            uint64_t qi = Q_LIMBS[i];
            for (int j = 0; j < N; j++) {
                ct.parts[0].data[i][j] = addmod(c1.parts[0].data[i][j], c2.parts[0].data[i][j], qi);
                ct.parts[1].data[i][j] = addmod(c1.parts[1].data[i][j], c2.parts[1].data[i][j], qi);
            }
        }
        ct.ntt_form = true; ct.scale = c1.scale;
        return ct;
    }
    // WE WILL DESTROY A TRILLION DOLLAR CORPORATION WITH CACHE LOCALITY AND COMPILER DISRESPECT
    //THE SILICON WILL OBEY
    Ciphertext multiply(const Ciphertext& ct1, const Ciphertext& ct2, const RelinKeyP& rlk) {
        double w = omp_get_wtime();
        size_t L = std::min(ct1.parts[0].data.size(), ct2.parts[0].data.size());
        size_t nQ = Q_LIMBS.size();
        uint64_t p_0 = P_LIMBS[0];

        //tensr: pure pointwise, no transforms
        std::vector<std::vector<uint64_t>> d0(L, std::vector<uint64_t>(N)),
                                        d1(L, std::vector<uint64_t>(N)),
                                        d2(L, std::vector<uint64_t>(N));
        #pragma omp parallel for
        for (int i = 0; i < (int)L; i++) {
            const Red& R = redQ[i];
            const auto &a0 = ct1.parts[0].data[i], &a1 = ct1.parts[1].data[i];
            const auto &b0 = ct2.parts[0].data[i], &b1 = ct2.parts[1].data[i];
            //LLVM WILL OBEY ME OR SO HELP ME GOD
            #pragma unroll 8
            for (int j = 0; j < N; j++) {
                d0[i][j] = red128((__uint128_t)a0[j] * b0[j], R);
                d2[i][j] = red128((__uint128_t)a1[j] * b1[j], R);
                d1[i][j] = red128((__uint128_t)a0[j] * b1[j] + (__uint128_t)a1[j] * b0[j], R);
            }
        }
        g_ph[0] += omp_get_wtime() - w; w = omp_get_wtime();
        
        //One intt per limb gets coefficients for centering
        std::vector<std::vector<uint64_t>> dig_ntt(L, std::vector<uint64_t>(N));
        std::vector<std::vector<int64_t>>  digits(L, std::vector<int64_t>(N));
        #pragma omp parallel for
        for (int k = 0; k < (int)L; k++) {
            uint64_t q_k = Q_LIMBS[k];
            uint64_t inv = q_hat_inv_mod_q[nQ - 1][k];
            for (int j = 0; j < N; j++) dig_ntt[k][j] = mulc(d2[k][j], inv, qhi_p[k], q_k);
            std::vector<uint64_t> tmp = dig_ntt[k];
            intt(tmp, *tabQ[k], q_k, N);
            for (int j = 0; j < N; j++)
                digits[k][j] = (tmp[j] > q_k / 2) ? (int64_t)tmp[j] - (int64_t)q_k : (int64_t)tmp[j];
        }
        g_ph[1] += omp_get_wtime() - w; w = omp_get_wtime();
        //key inner product, parallel over target limb
        std::vector<std::vector<uint64_t>> r0_Q(L), r1_Q(L);
        std::vector<uint64_t> r0_P, r1_P;

        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t <= (int)L; t++) {
            bool isP = (t == (int)L);
            uint64_t mod = isP ? p_0 : Q_LIMBS[t];
            const NTTTables& T = isP ? *tabP[0] : *tabQ[t];
            const Red& R = isP ? redP : redQ[t];

            static thread_local std::vector<std::vector<uint64_t>> LF;
            if (LF.size() < L) LF.resize(L);

            const uint64_t *lfp[16], *kbp[16], *kap[16];
            for (size_t k = 0; k < L; k++) {
                if (!isP && (size_t)t == k) {
                    lfp[k] = dig_ntt[k].data();               // already NTT form
                } else {
                    LF[k].resize(N);
                    for (int j = 0; j < N; j++) LF[k][j] = lift_signed(digits[k][j], mod);
                    ntt(LF[k], T, mod, N);
                    lfp[k] = LF[k].data();
                }
                kbp[k] = (isP ? rlk.b_P[k][0] : rlk.b_Q[k][t]).data();
                kap[k] = (isP ? rlk.a_P[k][0] : rlk.a_Q[k][t]).data();
            }

            std::vector<uint64_t> acc0(N), acc1(N);
            for (int j = 0; j < N; j++) {
                __uint128_t s0 = 0, s1 = 0;
                //YOU WILL PARRALELIZE OR YOU WILL SERIALIZE 6 FEET UNDER                    
                #pragma unroll 8
                for (size_t k = 0; k < L; k++) {
                    s0 += (__uint128_t)lfp[k][j] * kbp[k][j];
                    s1 += (__uint128_t)lfp[k][j] * kap[k][j];
                }
                acc0[j] = red128(s0, R);
                acc1[j] = red128(s1, R);
            }

            if (isP) {
                intt(acc0, T, mod, N);
                intt(acc1, T, mod, N);
                r0_P = std::move(acc0); r1_P = std::move(acc1);
            } else {
                r0_Q[t] = std::move(acc0); r1_Q[t] = std::move(acc1);
            }
        }
        g_ph[2] += omp_get_wtime() - w; w = omp_get_wtime();
        //mod-down in NTT form (in place)
        mod_down_ntt(r0_Q, r0_P, d0);
        mod_down_ntt(r1_Q, r1_P, d1);
        g_ph[3] += omp_get_wtime() - w; w = omp_get_wtime();
        Ciphertext out;
        out.parts.resize(2);
        out.parts[0].data = std::move(r0_Q);
        out.parts[1].data = std::move(r1_Q);
        out.scale = ct1.scale * ct2.scale;
        out.ntt_form = true;
        return out;
    }
    
    Ciphertext rescale(const Ciphertext& ct) {
        double w = omp_get_wtime();
        size_t L = ct.parts[0].data.size();
        if (L <= 1) throw std::runtime_error("Cannot rescale a 1-limb ciphertext!");
        size_t l = L - 1;
        uint64_t q_last = Q_LIMBS[l], q_last_half = q_last / 2;

        Ciphertext out;
        out.parts.resize(2);
        out.scale = ct.scale / (double)q_last;
        out.ntt_form = true;

        for (int p = 0; p < 2; p++) {
            out.parts[p].data.assign(l, std::vector<uint64_t>(N));
            std::vector<uint64_t> r = ct.parts[p].data[l];
            intt(r, *tabQ[l], q_last, N);

            #pragma omp parallel for
            for (int i = 0; i < (int)l; i++) {
                uint64_t qi = Q_LIMBS[i];
                uint64_t ql_mod = q_last >= qi ? q_last - qi : q_last;   // q_last mod qi
                const uint64_t c = q_inv_mod_q[l][i], cp = qinv_p[l][i];
                std::vector<uint64_t> A(N);
                for (int j = 0; j < N; j++) {
                    uint64_t rm = r[j] >= qi ? r[j] - qi : r[j];          // r < q_last < 2*qi
                    A[j] = (r[j] >= q_last_half) ? submod(rm, ql_mod, qi) : rm;
                }
                ntt(A, *tabQ[i], qi, N);
                for (int j = 0; j < N; j++)
                    out.parts[p].data[i][j] = mulc(submod(ct.parts[p].data[i][j], A[j], qi), c, cp, qi);
            }
        }
        g_ph[5] += omp_get_wtime() - w;
        return out;
    }

};
void print_vector(const std::vector<uint64_t> toPrint){
    for(int i=0;i<4;i++){
        std::cout << toPrint[i] << " ";
    }
    std::cout << "" << std::endl;
}
void print_vector(const std::vector<double> toPrint){
    for(int i=0;i<4;i++){
        std::cout << toPrint[i] << " ";
    }
    std::cout << "" << std::endl;
}
int main(){

    //setup sk
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<int> distrib(-1, 1);
    std::vector<int> s_key(N,0);
    for(int i=0;i<N;i++){
        s_key[i] = distrib(gen);
    }

    CKKSEngine engine;

    std::cout << "Setting up relin P tables" << std::endl;
    engine.setup_rns_tables(); 
    // for (int it = 0; it < 1000000; it++) {
    //     __uint128_t x = ((__uint128_t)gen() << 64 | gen()) >> (gen() % 5);   // up to ~2^124 range
    //     for (size_t i = 0; i < Q_LIMBS.size(); i++)
    //         if (red128(x, engine.redQ[i]) != (uint64_t)(x % Q_LIMBS[i])) { puts("red128 mismatch"); exit(1); }
    // }

    {
        std::vector<uint64_t> v(N);
        for (auto& x : v) x = gen() % Q_LIMBS[0];
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 200; r++) { ntt(v, *engine.tabQ[0], Q_LIMBS[0], N); intt(v, *engine.tabQ[0], Q_LIMBS[0], N); }
        double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
        printf("%.2f ns per butterfly\n", ns / (400.0 * 15 * (N / 2)));
    }
    {
        bool ok = true;
        for (size_t li = 0; li < Q_LIMBS.size() && ok; li++) {
            uint64_t q = Q_LIMBS[li];
            std::vector<uint64_t> v(N), s, w;
            for (auto& x : v) x = gen() % q;
            s = v; w = v;
            ntt_scalar(s, *engine.tabQ[li], q, N);
            ntt(w, *engine.tabQ[li], q, N);
            if (s != w) { printf("fwd mismatch limb %zu\n", li); ok = false; break; }
            intt_scalar(s, *engine.tabQ[li], q, N);
            intt(w, *engine.tabQ[li], q, N);
            if (s != w || w != v) { printf("inv mismatch limb %zu\n", li); ok = false; }
        }
        puts(ok ? "AVX512 NTT ok" : "AVX512 NTT FAILED");
    }
    std::cout << "Gen relin key." << std::endl;
    RelinKeyP rlk = engine.generate_relin_key(s_key);

    std::vector<uint64_t> cleartext0(N/2, 0);
    std::vector<uint64_t> cleartext1(N/2, 0);
    cleartext0[0] = 2; 
    cleartext1[0] = 5; 

    std::cout << "Encoding + Encrypting vectors" << std::endl;
    Plaintext pt0 = engine.encode(cleartext0);
    Plaintext pt1 = engine.encode(cleartext1);
    Ciphertext ct0 = engine.encrypt(pt0, s_key);
    Ciphertext ct1 = engine.encrypt(pt1, s_key);
    //calculate/benchamark based on instruictions per cycle
    // perf stat -e cycles,instructions,uops_executed.core,avx_insts.all ./your_benchmark
    std::cout << "Begining depth 10 circuit 5*2^10:" << std::endl;
    auto start = std::chrono::steady_clock::now();
    Ciphertext ct_mult = engine.multiply(ct0, ct1, rlk);    //1
    Ciphertext ct_rescaled = engine.rescale(ct_mult);
    ct_mult = engine.multiply(ct0, ct_rescaled, rlk); //2
    ct_rescaled = engine.rescale(ct_mult);
    ct_mult = engine.multiply(ct0, ct_rescaled, rlk);//3
    ct_rescaled = engine.rescale(ct_mult);
    ct_mult = engine.multiply(ct0, ct_rescaled, rlk);//4
    ct_rescaled = engine.rescale(ct_mult);
    ct_mult = engine.multiply(ct0, ct_rescaled, rlk);//5
    ct_rescaled = engine.rescale(ct_mult);
    ct_mult = engine.multiply(ct0, ct_rescaled, rlk);//6
    ct_rescaled = engine.rescale(ct_mult);
    ct_mult = engine.multiply(ct0, ct_rescaled, rlk);//7
    ct_rescaled = engine.rescale(ct_mult);
    ct_mult = engine.multiply(ct0, ct_rescaled, rlk);//8
    ct_rescaled = engine.rescale(ct_mult);
    ct_mult = engine.multiply(ct0, ct_rescaled, rlk);//9
    ct_rescaled = engine.rescale(ct_mult);
    ct_mult = engine.multiply(ct0, ct_rescaled, rlk);//10
    ct_rescaled = engine.rescale(ct_mult);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << elapsed.count() << " ms\n";
    //# plaintext ops is N/2 * 10
    std::cout << "Total number of plaintext operations = 10 * 16,384 = 163,840"<< std::endl;
    std::cout << "Total number of plaintext operations per second: " << 163840.0/(elapsed.count()/1000.0) << std::endl;
    for (int i = 0; i < 6; i++) printf("%-12s %.1f ms\n", g_phn[i], g_ph[i] * 1000);
    // std::cout << m_ops/(elapsed.count() /1000.0) << " million operations per second." <<std::endl;

    std::cout << "decrypting" << std::endl;
    Plaintext pt_mult = engine.decrypt(ct_rescaled, s_key);
    std::vector<double> result = engine.decode(pt_mult);
    
    std::cout << "Expected 5120, Result:   ";
    print_vector(result);

    return 0;
}
