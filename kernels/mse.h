#ifndef MSE_H
#define MSE_H

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

void lanzar_mse(float *d_batch, float *d_ref,
                float *d_rmse, int B, int H, int W);

#endif /* MSE_H */
