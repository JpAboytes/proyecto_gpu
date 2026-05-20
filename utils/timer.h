#ifndef TIMER_H
#define TIMER_H

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

typedef struct { cudaEvent_t start, stop; } GpuTimer;

void  timer_start(GpuTimer *t);
float timer_stop(GpuTimer *t);   /* devuelve milisegundos transcurridos */

#endif /* TIMER_H */
