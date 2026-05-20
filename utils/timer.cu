#include "timer.h"

void timer_start(GpuTimer *t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
    CUDA_CHECK(cudaEventRecord(t->start, 0));
}

/* Espera a que stop sea registrado, calcula el elapsed y destruye los eventos */
float timer_stop(GpuTimer *t) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(t->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    CUDA_CHECK(cudaEventDestroy(t->start));
    CUDA_CHECK(cudaEventDestroy(t->stop));
    return ms;
}
