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
const double DELTA = 1000000000; // 2^40;

const std::vector<uint64_t> Q_LIMBS = {
    1073741827, 1073742223, 1073742283, 1073742317, 1073742367, 
    1073742391, 1073742419, 1073742461, 1073742511, 1073742523
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
            uint64_t product = (A[i] * B[j]) % q;
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
            uint64_t mi = std::round((sum.real() / N) * DELTA);
            for (size_t k = 0; k < Q_LIMBS.size(); ++k) {
                pt.data[k][i] = wrap(mi, Q_LIMBS[k]);
            }
        }
        return pt;
    }
};
void print_vector(const std::vector<uint64_t> toPrint){
    for(int i=0;i<4;i++){
        std::cout << toPrint[i] << " " << std::endl;
    }
}
int main(){
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
    return 0;
}
