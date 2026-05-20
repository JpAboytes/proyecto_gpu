#include "grises.h"

/* Kernel dado por el profesor — convierte batch RGB (B,3,H,W) a grises (B,H,W) */
__global__ void escala_grises(float *entrada, float *salida, int B, int H, int W) {
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    if (fila < H && col < W) {
        for (int b = 0; b < B; b++) {
            int base = b * 3 * H * W;
            int i    = fila * W + col;
            salida[b*H*W + i] = 0.2989f * entrada[base + 0*H*W + i]
                               + 0.5870f * entrada[base + 1*H*W + i]
                               + 0.1140f * entrada[base + 2*H*W + i];
        }
    }
}

void lanzar_grises(float *d_entrada, float *d_salida, int B, int H, int W) {
    /* 16x16 = 256 hilos por bloque — multiplo de 32 (8 warps completos) */
    dim3 bloque(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    escala_grises<<<grid, bloque>>>(d_entrada, d_salida, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
