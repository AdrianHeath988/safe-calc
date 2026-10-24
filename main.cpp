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

const double PI = 3.14159265358979323846;
const int N = 1 << 3; 
const double DELTA = 1099511627776.0; //2^40

// 60-bit primes (congruent to 1 mod 2N for future NTT compatibility)
const std::vector<uint64_t> Q_LIMBS = {
    1152921504606584833ULL,
    1152921504598720513ULL,
    1152921504597016577ULL,
    1152921504595968001ULL,
    1152921504595640321ULL,
    1152921504593739777ULL,
    1152921504592822273ULL,
    1152921504589938689ULL,
    1152921504586530817ULL,
    1152921504580894721ULL
};
const std::vector<uint64_t> P_LIMBS = {
    1152921504550092801ULL, 1152921504546553857ULL, 1152921504545505281ULL,
    1152921504545243137ULL, 1152921504539869185ULL, 1152921504538558465ULL,
    1152921504535543809ULL, 1152921504532660225ULL, 1152921504531873793ULL,
    1152921504523354113ULL, 1152921504521388033ULL
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

std::vector<uint64_t> poly_mult_naive (const std::vector<uint64_t>& A, const std::vector<uint64_t>& B, uint64_t q){
    std::vector<__uint128_t> intermediate(2 * N -1, 0);
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            uint64_t product = (uint64_t)(((__uint128_t)A[i] * B[j]) % q);
            intermediate[i + j] = (intermediate[i+j]+product) % q;
        }
    }
    std::vector<uint64_t> out(N, 0);
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
inline uint64_t wrap(int64_t x, uint64_t q) {
    int64_t rem = x % (int64_t)q;
    return rem < 0 ? (uint64_t)(rem + q) : (uint64_t)rem;
}
struct RelinKeyP {
    std::vector<std::vector<uint64_t>> b_Q, a_Q; 
    std::vector<std::vector<uint64_t>> b_P, a_P; 
};
class CKKSEngine {
public:

    //setup phase for relin fast auxillirary bassi switch for later
    std::vector<uint64_t> q_hat_inv_mod_q, p_hat_inv_mod_p;
    std::vector<std::vector<uint64_t>> q_hat_mod_p, p_hat_mod_q;
    std::vector<uint64_t> Q_mod_p, P_mod_q, P_inv_mod_q;
    std::vector<std::vector<uint64_t>> q_inv_mod_q; // [limb_to_drop][remaining_limb]
    //precomputing a translation from the qi limbs to pi
    //use this so no crt necassary when doing eval steps (avoids big int)

    void setup_rns_tables() {
        mpz_class Q_val = 1; for (auto q : Q_LIMBS) Q_val *= q;
        mpz_class P_val = 1; for (auto p : P_LIMBS) P_val *= p;
        size_t nQ = Q_LIMBS.size(), nP = P_LIMBS.size();

        q_hat_inv_mod_q.resize(nQ); q_hat_mod_p.resize(nQ, std::vector<uint64_t>(nP)); Q_mod_p.resize(nP);
        p_hat_inv_mod_p.resize(nP); p_hat_mod_q.resize(nP, std::vector<uint64_t>(nQ)); P_mod_q.resize(nQ);
        P_inv_mod_q.resize(nQ);

        //q to p transition tables
        for (size_t i = 0; i < nQ; i++) {
            mpz_class Qi = Q_val / Q_LIMBS[i];
            mpz_class inv; mpz_invert(inv.get_mpz_t(), Qi.get_mpz_t(), mpz_class(Q_LIMBS[i]).get_mpz_t());
            q_hat_inv_mod_q[i] = inv.get_ui();
            
            for (size_t j = 0; j < nP; j++) {
                // Cast the expression template to mpz_class, then extract 64-bit int
                q_hat_mod_p[i][j] = mpz_class(Qi % P_LIMBS[j]).get_ui();
            }
        }
        for (size_t j = 0; j < nP; j++) {
            Q_mod_p[j] = mpz_class(Q_val % P_LIMBS[j]).get_ui();
        }

        //p -> q transition tables
        for (size_t k = 0; k < nP; k++) {
            mpz_class Pk = P_val / P_LIMBS[k];
            mpz_class inv; mpz_invert(inv.get_mpz_t(), Pk.get_mpz_t(), mpz_class(P_LIMBS[k]).get_mpz_t());
            p_hat_inv_mod_p[k] = inv.get_ui();
            
            for (size_t i = 0; i < nQ; i++) {
                p_hat_mod_q[k][i] = mpz_class(Pk % Q_LIMBS[i]).get_ui();
                // print("HERE 3");
            }
        }
        for (size_t i = 0; i < nQ; i++) {
            P_mod_q[i] = mpz_class(P_val % Q_LIMBS[i]).get_ui();
            mpz_class inv; mpz_invert(inv.get_mpz_t(), P_val.get_mpz_t(), mpz_class(Q_LIMBS[i]).get_mpz_t());
            P_inv_mod_q[i] = inv.get_ui();
        }


        //for rescale (inv of drop luimbs)
        q_inv_mod_q.resize(nQ, std::vector<uint64_t>(nQ));
        for (size_t l = 1; l < nQ; l++) {
            for (size_t i = 0; i < l; i++) {
                mpz_class inv; 
                mpz_invert(inv.get_mpz_t(), mpz_class(Q_LIMBS[l]).get_mpz_t(), mpz_class(Q_LIMBS[i]).get_mpz_t());
                q_inv_mod_q[l][i] = inv.get_ui(); 
            }
        }
    }
    //key is P * s^2
    //rewriting alot of enc algo (look at that first)
    RelinKeyP generate_relin_key(const std::vector<int>& secret) {
        RelinKeyP rlk;
        size_t nQ = Q_LIMBS.size(), nP = P_LIMBS.size();
        rlk.b_Q.resize(nQ, std::vector<uint64_t>(N)); rlk.a_Q.resize(nQ, std::vector<uint64_t>(N));
        rlk.b_P.resize(nP, std::vector<uint64_t>(N)); rlk.a_P.resize(nP, std::vector<uint64_t>(N));

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::normal_distribution<double> gaussian_dist(0.0, 1.78885);
        std::vector<int64_t> global_e(N);
        for(int j = 0; j < N; j++) global_e[j] = std::round(gaussian_dist(gen));

        std::vector<int64_t> s2(N, 0);
        for(int i = 0; i < N; ++i) {
            for(int j = 0; j < N; ++j) {
                int64_t prod = secret[i] * secret[j];
                if (i + j < N) s2[i + j] += prod;
                else s2[(i + j) % N] -= prod; 
            }
        }

        for(size_t i = 0; i < nQ; i++) {
            uint64_t mod = Q_LIMBS[i];
            std::uniform_int_distribution<uint64_t> dist(0, mod - 1);
            std::vector<uint64_t> a(N), s_mod(N), s2_mod(N);
            for(int j = 0; j < N; j++) {
                a[j] = dist(gen); s_mod[j] = wrap(secret[j], mod); s2_mod[j] = wrap(s2[j], mod);
            }
            std::vector<uint64_t> as = poly_mult_naive(a, s_mod, mod);
            for(int j = 0; j < N; j++) {
                uint64_t base = (mod - as[j]) % mod;
                base = (base + wrap(global_e[j], mod)) % mod;
                
                uint64_t ps2 = (uint64_t)(((__uint128_t)P_mod_q[i] * s2_mod[j]) % mod);
                rlk.b_Q[i][j] = (base + ps2) % mod;
                rlk.a_Q[i][j] = a[j];
            }
        }
        //now that we have everything, make the key
        for(size_t i = 0; i < nP; i++) {
            uint64_t mod = P_LIMBS[i];
            std::uniform_int_distribution<uint64_t> dist(0, mod - 1);
            std::vector<uint64_t> a(N), s_mod(N);
            for(int j = 0; j < N; j++) {
                a[j] = dist(gen); s_mod[j] = wrap(secret[j], mod);
            }
            std::vector<uint64_t> as = poly_mult_naive(a, s_mod, mod);
            for(int j = 0; j < N; j++) {
                uint64_t base = (mod - as[j]) % mod;
                rlk.b_P[i][j] = (base + wrap(global_e[j], mod)) % mod;
                rlk.a_P[i][j] = a[j];
            }
        }
        // printf("end of relin");
        return rlk;
    }
    //extend to QUP basis using precomputed swithing
    //do this so later we can use AVX simd
    std::vector<std::vector<uint64_t>> fast_mod_up(const std::vector<std::vector<uint64_t>>& d_Q) {
        size_t nQ = Q_LIMBS.size(), nP = P_LIMBS.size();
        std::vector<std::vector<uint64_t>> d_P(nP, std::vector<uint64_t>(N, 0));

        for (int j = 0; j < N; j++) {
            uint64_t sum_frac = 0, v = 0;
            std::vector<uint64_t> x_prime(nQ);
            for (size_t i = 0; i < nQ; i++) {
                uint64_t qi = Q_LIMBS[i];
                x_prime[i] = (uint64_t)(((__uint128_t)d_Q[i][j] * q_hat_inv_mod_q[i]) % qi);
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
                for (size_t i = 0; i < nQ; i++) sum += (__uint128_t)x_prime[i] * q_hat_mod_p[i][k]; 
                uint64_t res = sum % pk;
                uint64_t sub = ((__uint128_t)v * Q_mod_p[k]) % pk;
                d_P[k][j] = (res + pk - sub) % pk;
            }
        }
        return d_P;
    }

    std::vector<std::vector<uint64_t>> fast_mod_down(
        const std::vector<std::vector<uint64_t>>& c_Q, const std::vector<std::vector<uint64_t>>& c_P) {
        size_t nQ = Q_LIMBS.size(), nP = P_LIMBS.size();
        std::vector<std::vector<uint64_t>> res_Q(nQ, std::vector<uint64_t>(N, 0));

        for (int j = 0; j < N; j++) {
            uint64_t sum_frac = 0, v = 0;
            std::vector<uint64_t> x_prime(nP);
            for (size_t k = 0; k < nP; k++) {
                uint64_t pk = P_LIMBS[k];
                x_prime[k] = (uint64_t)(((__uint128_t)c_P[k][j] * p_hat_inv_mod_p[k]) % pk);
                uint64_t frac = (uint64_t)(((__uint128_t)x_prime[k] << 64) / pk);
                uint64_t old = sum_frac;
                sum_frac += frac;
                if (sum_frac < old) v++;
            }
            uint64_t old = sum_frac;
            sum_frac += (1ULL << 63); 
            if (sum_frac < old) v++;

            for (size_t i = 0; i < nQ; i++) {
                uint64_t qi = Q_LIMBS[i];
                __uint128_t sum = 0;
                for (size_t k = 0; k < nP; k++) sum += (__uint128_t)x_prime[k] * p_hat_mod_q[k][i];
                uint64_t r_mod_qi = sum % qi;
                uint64_t sub = ((__uint128_t)v * P_mod_q[i]) % qi;
                r_mod_qi = (r_mod_qi + qi - sub) % qi;

                uint64_t diff = (c_Q[i][j] + qi - r_mod_qi) % qi;
                res_Q[i][j] = (uint64_t)(((__uint128_t)diff * P_inv_mod_q[i]) % qi);
            }
        }
        return res_Q;
    }
    Plaintext encode(std::vector<uint64_t>& vals){
        //start by extending to forward ordered hermitian vector
        std::vector<std::complex<double>> y(N);   //expect n/2 vals
        for (int j = 0; j < N / 2; ++j) {
            y[j] = std::complex<double>(vals[j], 0.0);
            y[j+N/2] = conj(y[j]); 
        }
        
        
        

        
        //primitive 2N root of unity. every other root in vandermonde is just an odd power of this
        std::complex<double> zeta(cos(PI / N), sin(PI / N));

        //get the rest of the roots
        std::vector<std::complex<double>> roots(N);
        //compute first row of W squiggle. rest are just powers
        for(int i=0;i<N/2;i++){
            roots[i] = std::pow(zeta, 2 * i + 1);
            roots[i + N / 2] = std::conj(roots[i]);
        }
        //get W squiggle
        std::vector<std::vector<std::complex<double>>> w_matrix(N, std::vector<std::complex<double>>(N, 0));;
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                w_matrix[i][j] = std::pow(std::conj(roots[j]), i);
            }
        }

        //now compute I * W * v / n. 
        //instead of actually doing mult by I, we just ignore the parts we dont need
        Plaintext pt;  
        for(int i=0;i<N;i++){
            std::complex<double> sum(0.0, 0.0);
            //be a bit nice to the cache :)
            for (int j = 0; j < N; ++j) {
                sum += w_matrix[i][j] * y[j];
            }
            //vals in the front, conj in the back
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
            std::vector<uint64_t> as = poly_mult_naive(a, sk_limb, qi);

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
            std::vector<uint64_t> c1s = poly_mult_naive(ct.parts[1].data[i], sk_limb, qi);
            
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
        size_t nQ = Q_LIMBS.size(), nP = P_LIMBS.size();
        //mult naive for now
        std::vector<std::vector<uint64_t>> d0(nQ), d1(nQ), d2(nQ);
        for (size_t i = 0; i < nQ; i++) {
            uint64_t qi = Q_LIMBS[i];
            d0[i] = poly_mult_naive(ct1.parts[0].data[i], ct2.parts[0].data[i], qi);
            d2[i] = poly_mult_naive(ct1.parts[1].data[i], ct2.parts[1].data[i], qi);
            
            auto cross1 = poly_mult_naive(ct1.parts[0].data[i], ct2.parts[1].data[i], qi);
            auto cross2 = poly_mult_naive(ct1.parts[1].data[i], ct2.parts[0].data[i], qi);
            
            d1[i].resize(N);
            for (int j = 0; j < N; j++) d1[i][j] = (cross1[j] + cross2[j]) % qi;
        }

        std::vector<std::vector<uint64_t>> d2_P = fast_mod_up(d2);
        //extend to QUP basis and do relin
        std::vector<std::vector<uint64_t>> r0_Q(nQ), r1_Q(nQ), r0_P(nP), r1_P(nP);
        for (size_t i = 0; i < nQ; i++) {
            r0_Q[i] = poly_mult_naive(d2[i], rlk.b_Q[i], Q_LIMBS[i]);
            r1_Q[i] = poly_mult_naive(d2[i], rlk.a_Q[i], Q_LIMBS[i]);
        }
        for (size_t k = 0; k < nP; k++) {
            r0_P[k] = poly_mult_naive(d2_P[k], rlk.b_P[k], P_LIMBS[k]);
            r1_P[k] = poly_mult_naive(d2_P[k], rlk.a_P[k], P_LIMBS[k]);
        }

        auto relin_c0 = fast_mod_down(r0_Q, r0_P);
        auto relin_c1 = fast_mod_down(r1_Q, r1_P);

        Ciphertext ct_out;
        ct_out.parts.resize(2, Plaintext());
        
        for (size_t i = 0; i < nQ; i++) {
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
    cleartext0[1] = 3;
    cleartext1[0] = 4; 
    cleartext1[1] = 5;

    std::cout << "Encoding + Encrypting vectors" << std::endl;
    Plaintext pt0 = engine.encode(cleartext0);
    Plaintext pt1 = engine.encode(cleartext1);
    Ciphertext ct0 = engine.encrypt(pt0, s_key);
    Ciphertext ct1 = engine.encrypt(pt1, s_key);

    std::cout << "mult + relin" << std::endl;
    Ciphertext ct_mult = engine.multiply(ct0, ct1, rlk);
    std::cout << "Rescaling to manage noise and scale" << std::endl;
    Ciphertext ct_rescaled = engine.rescale(ct_mult);
    std::cout << "decrypting" << std::endl;
    Plaintext pt_mult = engine.decrypt(ct_rescaled, s_key);
    std::vector<double> result = engine.decode(pt_mult);
    
    std::cout << "\nExpected Result: 8 15 0 0" << std::endl;
    std::cout << "Actual Result:   ";
    print_vector(result);

    return 0;
}
