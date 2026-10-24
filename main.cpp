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

struct Plaintext {
    std::vector<std::vector<uint64_t>> data;
    Plaintext() : data(Q_LIMBS.size(), std::vector<uint64_t>(N, 0)){}; // init blank pt to have Q_LIMBS moduli, each modulus having all 0s
};
struct Ciphertext {
    std::vector<Plaintext> parts;
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
class CKKSEngine {
public:
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
    std::vector<uint64_t> decode(const Plaintext& pt) {
        //crt combine
        //gmp bigint
        mpz_class Q = 1;
        for(uint64_t q : Q_LIMBS) {
            Q *= q;
        }
        //shift lower half into negatives
        mpz_class Q_half = Q / 2;

        std::vector<mpz_class> m_big(N, 0);
        
        //recontrsuct
        for(size_t i = 0; i < Q_LIMBS.size(); ++i) {
            mpz_class qi = Q_LIMBS[i];
            mpz_class Mi = Q / qi;
            
            //GMP Modulo Inverse: yi = Mi^(-1) mod qi
            mpz_class yi;
            mpz_invert(yi.get_mpz_t(), Mi.get_mpz_t(), qi.get_mpz_t());
            
            for(int j = 0; j < N; ++j) {
                mpz_class term = pt.data[i][j];
                term = (term * Mi * yi) % Q;
                m_big[j] = (m_big[j] + term) % Q;
            }
        }

        //Undo the wrap
        std::vector<int64_t> m(N);
        for(int j = 0; j < N; ++j) {
            if (m_big[j] > Q_half) {
                m_big[j] -= Q;
            }
            // The un-rns coefficient is now small enough
            m[j] = std::stoll(m_big[j].get_str()); 
        }

        //evaluating the polynomial at the roots (mult by W* squiggle)
        std::vector<uint64_t> res(N / 2);
        std::complex<double> zeta(cos(PI / N), sin(PI / N));
        
        //only evaluate at the first N/2 roots because those hold our original real values
        for (int j = 0; j < N / 2; ++j) {
            std::complex<double> root = std::pow(zeta, 2 * j + 1);
            std::complex<double> evaluated(0.0, 0.0);
            
            // sum( m[k] * root^k )
            for (int k = 0; k < N; ++k) {
                std::complex<double> root_k = std::pow(root, k);
                evaluated += (double)m[k] * root_k;
            }
            
            res[j] = std::round(evaluated.real() / DELTA);
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
        return ct;
    }
    Plaintext decrypt(const Ciphertext& ct, const std::vector<int>& secret){
        Plaintext pt;
        
        for (size_t i = 0; i < Q_LIMBS.size(); ++i) {
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
        return pt;
    }
};
void print_vector(const std::vector<uint64_t> toPrint){
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



    std::vector<uint64_t> cleartext0(N/2, 0);
    std::vector<uint64_t> cleartext1(N/2, 0);
    cleartext0[0] = 1;
    cleartext0[1] = 2;
    cleartext1[0] = 1;
    cleartext1[1] = 2;
    // print_vector(cleartext0);
    // print_vector(cleartext1);
    CKKSEngine engine;
    Plaintext pt = engine.encode(cleartext0);
    print_vector(pt.data[0]);
    Ciphertext ct = engine.encrypt(pt, s_key);
    std::cout << "\nEncrypted c0 (Limb 0): " << std::endl;
    print_vector(ct.parts[0].data[0]);
    pt = engine.decrypt(ct, s_key);
    std::vector<uint64_t> result = engine.decode(pt);
    std::cout << "\n Result: " << std::endl;
    print_vector(result);
    return 0;
}
