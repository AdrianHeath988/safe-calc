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
};
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
    std::vector<uint64_t> forward_twiddles;
    std::vector<uint64_t> inverse_twiddles;
    uint64_t n_inv;
};

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
//Cooley-Tukey Forward NTT
void ntt(std::vector<uint64_t>& a, const std::vector<uint64_t>& twiddles, uint64_t q, int N) {
    int t = N;
    for (int m = 1; m < N; m <<= 1) {
        t >>= 1;
        for (int i = 0; i < m; i++) {
            int j1 = 2 * i * t;
            int j2 = j1 + t - 1;
            uint64_t W = twiddles[m + i];
            for (int j = j1; j <= j2; j++) {
                uint64_t U = a[j];
                uint64_t V = (uint64_t)(((__uint128_t)a[j + t] * W) % q);
                a[j] = (U + V) % q;
                a[j + t] = (U + q - V) % q;
            }
        }
    }
}

//Gentleman-Sande Inverse NTT
void intt(std::vector<uint64_t>& a, const std::vector<uint64_t>& inv_twiddles, uint64_t q, int N, uint64_t n_inv) {
    int t = 1;
    for (int m = N; m > 1; m >>= 1) {
        int j1 = 0;
        int h = m >> 1;
        for (int i = 0; i < h; i++) {
            int j2 = j1 + t - 1;
            uint64_t W = inv_twiddles[h + i];
            for (int j = j1; j <= j2; j++) {
                uint64_t U = a[j];
                uint64_t V = a[j + t];
                a[j] = (U + V) % q;
                a[j + t] = (uint64_t)(((__uint128_t)(U + q - V) * W) % q);
            }
            j1 += 2 * t;
        }
        t <<= 1;
    }
    //normalization
    for (int i = 0; i < N; i++) {
        a[i] = (uint64_t)(((__uint128_t)a[i] * n_inv) % q);
    }
}

std::vector<uint64_t> ntt_mul(const std::vector<uint64_t>& A, const std::vector<uint64_t>& B, uint64_t q) {
    int N = A.size();
    
    //cache twiddles per prime to guarantee O(N log N) speed on repeat calls
    static std::unordered_map<uint64_t, NTTTables> cache;
    const NTTTables& tables = get_tables(q, N);

    //mutable copies
    std::vector<uint64_t> a_ntt = A;
    std::vector<uint64_t> b_ntt = B;
    
    ntt(a_ntt, tables.forward_twiddles, q, N);
    ntt(b_ntt, tables.forward_twiddles, q, N);
    
    std::vector<uint64_t> c_ntt(N);
    
    //Pointwise multiplication 
    //(A and B are in bit-reversed order, which is perfectly fine!)
    for(int i = 0; i < N; i++) {
        c_ntt[i] = (uint64_t)(((__uint128_t)a_ntt[i] * b_ntt[i]) % q);
    }
    
    intt(c_ntt, tables.inverse_twiddles, q, N, tables.n_inv);
    
    return c_ntt;
}
inline uint64_t wrap(int64_t x, uint64_t q) {
    int64_t rem = x % (int64_t)q;
    return rem < 0 ? (uint64_t)(rem + q) : (uint64_t)rem;
}
class CKKSEngine {
public:

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
                ntt(base_vec, tables.forward_twiddles, mod, N);
                ntt(a, tables.forward_twiddles, mod, N);
                
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
                ntt(base_vec, tables.forward_twiddles, mod, N);
                ntt(a, tables.forward_twiddles, mod, N);
                
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
        }
        ct.scale = pt.scale;
        return ct;
    }
    Plaintext decrypt(const Ciphertext& ct, const std::vector<int>& secret){
        Plaintext pt;
        size_t current_limbs = ct.parts[0].data.size();
        pt.data.resize(current_limbs, std::vector<uint64_t>(N, 0));
        for (size_t i = 0; i < current_limbs; ++i) {
            uint64_t qi = Q_LIMBS[i];
            
            //project secret key into field
            std::vector<uint64_t> sk_limb(N);
            for (int j = 0; j < N; ++j) {
                sk_limb[j] = wrap(secret[j], qi);
            }
            
            //compute c1 * s mod qi
            std::vector<uint64_t> c1s = ntt_mul(ct.parts[1].data[i], sk_limb, qi);
            
            //compute m = c0 + c1*s mod qi
            for (int j = 0; j < N; ++j) {
                pt.data[i][j] = (ct.parts[0].data[i][j] + c1s[j]) % qi;
            }
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
            for(int j=0;j<N;j++){
                uint64_t intermediateA = c1.parts[0].data[i][j]+ c2.parts[0].data[i][j];
                uint64_t intermediateB = c1.parts[1].data[i][j]+ c2.parts[1].data[i][j];
                
                ct.parts[0].data[i][j] = (intermediateA >= qi) ? (intermediateA - qi) : intermediateA;
                ct.parts[1].data[i][j] = (intermediateB >= qi) ? (intermediateB - qi) : intermediateB;
            }
        }
        return ct;
    }
    Ciphertext multiply(const Ciphertext& ct1, const Ciphertext& ct2, const RelinKeyP& rlk) {
        size_t current_limbs = std::min(ct1.parts[0].data.size(), ct2.parts[0].data.size());
        size_t nP = P_LIMBS.size();
        //mult naive for now
        std::vector<std::vector<uint64_t>> d0(current_limbs), d1(current_limbs), d2(current_limbs);
        #pragma omp parallel for
        for (size_t i = 0; i < current_limbs; i++) {
            uint64_t qi = Q_LIMBS[i];
            d0[i] = ntt_mul(ct1.parts[0].data[i], ct2.parts[0].data[i], qi);
            d2[i] = ntt_mul(ct1.parts[1].data[i], ct2.parts[1].data[i], qi);
            
            auto cross1 = ntt_mul(ct1.parts[0].data[i], ct2.parts[1].data[i], qi);
            auto cross2 = ntt_mul(ct1.parts[1].data[i], ct2.parts[0].data[i], qi);
            
            d1[i].resize(N);
            for (int j = 0; j < N; j++) d1[i][j] = (cross1[j] + cross2[j]) % qi;
        }

        size_t nQ = Q_LIMBS.size();
        uint64_t p_0 = P_LIMBS[0];

        //Gadget digits: signed, centered mod q_k, computed once
        std::vector<std::vector<int64_t>> digits(current_limbs, std::vector<int64_t>(N));
        for (size_t k = 0; k < current_limbs; k++) {
            uint64_t q_k = Q_LIMBS[k];
            uint64_t inv = q_hat_inv_mod_q[nQ - 1][k];
            for (int j = 0; j < N; j++) {
                uint64_t v = (uint64_t)(((__uint128_t)d2[k][j] * inv) % q_k);
                digits[k][j] = (v > q_k / 2) ? (int64_t)v - (int64_t)q_k : (int64_t)v;
            }
        }

        //Inner product with the key, parallel over target limb 
        std::vector<std::vector<uint64_t>> r0_Q(current_limbs), r1_Q(current_limbs);
        std::vector<uint64_t> r0_P, r1_P;

        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t <= (int)current_limbs; t++) {
            bool isP = (t == (int)current_limbs);
            uint64_t mod = isP ? p_0 : Q_LIMBS[t];
            const NTTTables& T = get_tables(mod, N);

            std::vector<uint64_t> acc0(N, 0), acc1(N, 0), lifted(N);
            for (size_t k = 0; k < current_limbs; k++) {
                for (int j = 0; j < N; j++) lifted[j] = wrap(digits[k][j], mod);
                ntt(lifted, T.forward_twiddles, mod, N);          // NTT only the digit

                const auto& kb = isP ? rlk.b_P[k][0] : rlk.b_Q[k][t];  //already NTT domain
                const auto& ka = isP ? rlk.a_P[k][0] : rlk.a_Q[k][t];
                for (int j = 0; j < N; j++) {
                    acc0[j] = (acc0[j] + (uint64_t)(((__uint128_t)lifted[j] * kb[j]) % mod)) % mod;
                    acc1[j] = (acc1[j] + (uint64_t)(((__uint128_t)lifted[j] * ka[j]) % mod)) % mod;
                }
            }
            intt(acc0, T.inverse_twiddles, mod, N, T.n_inv);      //back to coefficients BEFORE mod-down
            intt(acc1, T.inverse_twiddles, mod, N, T.n_inv);

            if (isP) { r0_P = std::move(acc0); r1_P = std::move(acc1); }
            else     { r0_Q[t] = std::move(acc0); r1_Q[t] = std::move(acc1); }
        }

        auto relin_c0 = fast_mod_down(r0_Q, r0_P);
        auto relin_c1 = fast_mod_down(r1_Q, r1_P);
        Ciphertext ct_out;
        ct_out.parts.resize(2, Plaintext());
        ct_out.parts[0].data.resize(current_limbs, std::vector<uint64_t>(N, 0));
        ct_out.parts[1].data.resize(current_limbs, std::vector<uint64_t>(N, 0));
        
        //last Tensor Accumulation
        #pragma omp parallel for
        for (size_t i = 0; i < current_limbs; i++) {
            uint64_t qi = Q_LIMBS[i];
            for (int j = 0; j < N; j++) {
                ct_out.parts[0].data[i][j] = (d0[i][j] + relin_c0[i][j]) % qi;
                ct_out.parts[1].data[i][j] = (d1[i][j] + relin_c1[i][j]) % qi;
            }
        }
        ct_out.scale = ct1.scale * ct2.scale;
        return ct_out;
    }
    Ciphertext rescale(const Ciphertext& ct) {
        Ciphertext ct_out;
        size_t current_limbs = ct.parts[0].data.size();
        if (current_limbs <= 1) throw std::runtime_error("Cannot rescale a 1-limb ciphertext!");

        size_t l = current_limbs - 1; //index of the limb being dropped
        uint64_t q_last = Q_LIMBS[l];
        uint64_t q_last_half = q_last / 2;

        ct_out.parts.resize(2, Plaintext());
        ct_out.parts[0].data.resize(l, std::vector<uint64_t>(N));
        ct_out.parts[1].data.resize(l, std::vector<uint64_t>(N));
        
        //scale reduction
        ct_out.scale = ct.scale / (double)q_last;

        for (int p = 0; p < 2; p++) {   //each part of the ct
            #pragma omp parallel for
            for (int j = 0; j < N; j++) {
                uint64_t r = ct.parts[p].data[l][j];
                bool round_up = (r >= q_last_half);

                for (size_t i = 0; i < l; i++) {
                    uint64_t qi = Q_LIMBS[i];
                    uint64_t A_i;
                    
                    if (!round_up) {
                        A_i = r % qi;
                    } else {
                        //(r - q_last) mod qi
                        uint64_t q_last_mod_qi = q_last % qi;
                        A_i = (r % qi + qi - q_last_mod_qi) % qi;
                    }

                    //sub centered remainder and multiply by inverse
                    uint64_t diff = (ct.parts[p].data[i][j] + qi - A_i) % qi;
                    ct_out.parts[p].data[i][j] = (uint64_t)(((__uint128_t)diff * q_inv_mod_q[l][i]) % qi);
                }
            }
        }
        return ct_out;
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
    // std::cout << m_ops/(elapsed.count() /1000.0) << " million operations per second." <<std::endl;

    std::cout << "decrypting" << std::endl;
    Plaintext pt_mult = engine.decrypt(ct_rescaled, s_key);
    std::vector<double> result = engine.decode(pt_mult);
    
    std::cout << "Expected 5120, Result:   ";
    print_vector(result);

    return 0;
}
