#ifndef NORMALIZAR_H
#define NORMALIZAR_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) {                                         \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error en %s:%d \xe2\x80\x94 %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
}
#endif

void lanzar_normalizar(float *d_entrada, float *d_salida,
                       float *d_maximos, int B, int H, int W);

#endif /* NORMALIZAR_H */
