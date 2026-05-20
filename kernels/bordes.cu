#include "bordes.h"

/* Filtro Sobel sobre batch de imagenes en escala de grises (B,H,W) */
__global__ void filtro_sobel(float *entrada, float *salida, int B, int H, int W) {
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;

    if (fila < H && col < W) {
        for (int b = 0; b < B; b++) {
            /* Pixeles en el borde de la imagen -> magnitud 0 (sin vecindad completa) */
            if (fila == 0 || fila == H - 1 || col == 0 || col == W - 1) {
                salida[b * H * W + fila * W + col] = 0.0f;
                continue;
            }

            /* Vecindad 3x3 centrada en (fila, col) */
            float p00 = entrada[b*H*W + (fila-1)*W + (col-1)];
            float p01 = entrada[b*H*W + (fila-1)*W + (col  )];
            float p02 = entrada[b*H*W + (fila-1)*W + (col+1)];
            float p10 = entrada[b*H*W + (fila  )*W + (col-1)];
            float p12 = entrada[b*H*W + (fila  )*W + (col+1)];
            float p20 = entrada[b*H*W + (fila+1)*W + (col-1)];
            float p21 = entrada[b*H*W + (fila+1)*W + (col  )];
            float p22 = entrada[b*H*W + (fila+1)*W + (col+1)];

            /* Gx: [[-1,0,+1],[-2,0,+2],[-1,0,+1]] */
            float gx = -p00 + p02
                       - 2.0f*p10 + 2.0f*p12
                       - p20 + p22;

            /* Gy: [[-1,-2,-1],[0,0,0],[+1,+2,+1]] */
            float gy = -p00 - 2.0f*p01 - p02
                       + p20 + 2.0f*p21 + p22;

            salida[b * H * W + fila * W + col] = sqrtf(gx*gx + gy*gy);
        }
    }
}

void lanzar_bordes(float *d_entrada, float *d_salida, int B, int H, int W) {
    /* 16x16 = 256 hilos — multiplo de 32 (warp size) */
    dim3 bloque(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    filtro_sobel<<<grid, bloque>>>(d_entrada, d_salida, B, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
