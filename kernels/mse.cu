#include "mse.h"

/* Calcula RMSE entre cada imagen del batch y una imagen de referencia.
   Grid: dim3(B) — un bloque por imagen
   Block: dim3(256) — 8 warps, multiplo de 32
   Reduccion en arbol con shared memory de tamano fijo 256               */
__global__ void kernel_mse(float *d_batch, float *d_ref, float *d_rmse,
                           int B, int H, int W) {
    __shared__ float sdata[256];

    int b   = blockIdx.x;   /* cada bloque calcula el RMSE de una imagen */
    int tid = threadIdx.x;
    int N   = H * W;

    /* Acumula suma parcial de errores cuadraticos con stride = blockDim.x */
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = d_batch[b * N + i] - d_ref[i];
        local_sum += diff * diff;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    /* Reduccion en arbol estandar: stride = blockDim.x/2, stride >>= 1 */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        d_rmse[b] = sqrtf(sdata[0] / (float)N);
}

void lanzar_mse(float *d_batch, float *d_ref,
                float *d_rmse, int B, int H, int W) {
    if (B == 0) return;
    /* Un bloque de 256 hilos (8 warps) por imagen del batch */
    kernel_mse<<<B, 256>>>(d_batch, d_ref, d_rmse, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
