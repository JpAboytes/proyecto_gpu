#ifndef BORDES_H
#define BORDES_H

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

void lanzar_bordes(float *d_entrada, float *d_salida, int B, int H, int W);

#endif /* BORDES_H */
