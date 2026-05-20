#include "normalizar.h"

/* ── PASO A: reduce el maximo de cada imagen con shared memory ─────────────
   Grid: dim3(B) — un bloque por imagen
   Block: dim3(256) — 8 warps, multiplo de 32                               */
__global__ void kernel_reduce_max(float *d_entrada, float *d_maximos,
                                  int B, int H, int W) {
    extern __shared__ float sdata[];

    int b   = blockIdx.x;   /* cada bloque maneja una imagen */
    int tid = threadIdx.x;
    int N   = H * W;

    /* Cada hilo recorre sus elementos con stride = blockDim.x y guarda el maximo local */
    float local_max = -1e38f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = d_entrada[b * N + i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();

    /* Reduccion en arbol */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && sdata[tid + stride] > sdata[tid])
            sdata[tid] = sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) d_maximos[b] = sdata[0];
}

/* ── PASO B: divide cada pixel por el maximo de su imagen ─────────────────
   Grid: 2D cubre H x W; loop interno sobre B
   Block: dim3(16,16) — 256 hilos, multiplo de 32
   Soporta operacion in-place (d_salida == d_entrada)                        */
__global__ void kernel_dividir(float *d_entrada, float *d_salida,
                               float *d_maximos, int B, int H, int W) {
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila < H && col < W) {
        for (int b = 0; b < B; b++) {
            int   idx = b * H * W + fila * W + col;
            float mx  = d_maximos[b];
            /* Evitar division por cero: si maximo es insignificante, dejar valor */
            d_salida[idx] = (mx < 1e-8f) ? d_entrada[idx] : d_entrada[idx] / mx;
        }
    }
}

void lanzar_normalizar(float *d_entrada, float *d_salida,
                       float *d_maximos, int B, int H, int W) {
    if (B == 0) return;

    /* PASO A — 256 hilos (8 warps), shared mem = 256 floats por bloque */
    kernel_reduce_max<<<B, 256, 256 * sizeof(float)>>>(d_entrada, d_maximos, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* PASO B — grid 2D, loop interno sobre B */
    dim3 bloque(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    kernel_dividir<<<grid, bloque>>>(d_entrada, d_salida, d_maximos, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
