#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "kernels/grises.h"
#include "kernels/bordes.h"
#include "kernels/normalizar.h"
#include "kernels/mse.h"
#include "utils/imagen.h"
#include "utils/timer.h"

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

int main(void) {

    /* ── 1. Cargar imagenes de imagenes/ ──────────────────────────────── */
    int B = 0, H = 0, W = 0;
    float *h_rgb = cargar_imagenes("imagenes", &B, &H, &W);
    printf("Imagenes cargadas: %d  |  Tamano: %d x %d\n\n", B, H, W);

    /* ── 2. cudaMalloc ────────────────────────────────────────────────── */
    float *d_rgb, *d_gris, *d_bordes, *d_norm, *d_maximos, *d_rmse, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_rgb,     (size_t)B * 3 * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gris,    (size_t)B     * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bordes,  (size_t)B     * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm,    (size_t)B     * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maximos, (size_t)B             * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rmse,    (size_t)B             * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref,     (size_t)      H * W   * sizeof(float)));

    GpuTimer t;

    /* ── 3. Transferencia H->D del batch RGB ──────────────────────────── */
    timer_start(&t);
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb,
                          (size_t)B * 3 * H * W * sizeof(float),
                          cudaMemcpyHostToDevice));
    float ms_htod = timer_stop(&t);
    printf("[H->D RGB]        %.4f ms\n", ms_htod);

    /* ── 4. K1: escala de grises ──────────────────────────────────────── */
    timer_start(&t);
    lanzar_grises(d_rgb, d_gris, B, H, W);
    float ms_grises = timer_stop(&t);
    printf("[K1 Grises]       %.4f ms\n", ms_grises);

    /* Referencia: primera imagen en grises — copia D->D, sin bajar a CPU */
    CUDA_CHECK(cudaMemcpy(d_ref, d_gris,
                          (size_t)H * W * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    /* ── 5. K2: deteccion de bordes (Sobel) ──────────────────────────── */
    timer_start(&t);
    lanzar_bordes(d_gris, d_bordes, B, H, W);
    float ms_bordes = timer_stop(&t);
    printf("[K2 Bordes]       %.4f ms\n", ms_bordes);

    /* ── 6. K3: normalizacion por maximo ──────────────────────────────── */
    timer_start(&t);
    lanzar_normalizar(d_bordes, d_norm, d_maximos, B, H, W);
    float ms_norm = timer_stop(&t);
    printf("[K3 Normalizar]   %.4f ms\n", ms_norm);

    /* ── 7. K4: RMSE respecto a la referencia ─────────────────────────── */
    timer_start(&t);
    lanzar_mse(d_norm, d_ref, d_rmse, B, H, W);
    float ms_mse = timer_stop(&t);
    printf("[K4 MSE]          %.4f ms\n", ms_mse);

    /* ── 8. Transferencia D->H del vector RMSE ────────────────────────── */
    float *h_rmse = (float*)malloc((size_t)B * sizeof(float));
    timer_start(&t);
    CUDA_CHECK(cudaMemcpy(h_rmse, d_rmse,
                          (size_t)B * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float ms_dtoh = timer_stop(&t);
    printf("[D->H RMSE]       %.4f ms\n\n", ms_dtoh);

    /* ── 9. Tabla de tiempos y RMSE por imagen ────────────────────────── */
    printf("=========================================\n");
    printf("%-24s %12s\n", "Operacion",       "Tiempo (ms)");
    printf("-----------------------------------------\n");
    printf("%-24s %12.4f\n", "Transferencia H->D", ms_htod);
    printf("%-24s %12.4f\n", "K1 Grises",          ms_grises);
    printf("%-24s %12.4f\n", "K2 Bordes (Sobel)",  ms_bordes);
    printf("%-24s %12.4f\n", "K3 Normalizar",      ms_norm);
    printf("%-24s %12.4f\n", "K4 MSE",             ms_mse);
    printf("%-24s %12.4f\n", "Transferencia D->H", ms_dtoh);
    printf("=========================================\n\n");

    printf("RMSE por imagen (referencia = imagen 0 en grises):\n");
    for (int b = 0; b < B; b++)
        printf("  [%3d]  %.6f\n", b, h_rmse[b]);
    printf("\n");

    /* ── 10. Bajar imagenes intermedias a CPU y guardar PNGs ──────────── */
    float *h_rgb0    = (float*)malloc((size_t)3 * H * W * sizeof(float));
    float *h_gris0   = (float*)malloc((size_t)    H * W * sizeof(float));
    float *h_bordes0 = (float*)malloc((size_t)    H * W * sizeof(float));
    float *h_norm0   = (float*)malloc((size_t)    H * W * sizeof(float));
    char nombre[128];

    for (int b = 0; b < B; b++) {
        CUDA_CHECK(cudaMemcpy(h_rgb0, d_rgb + (size_t)b * 3 * H * W,
                              (size_t)3 * H * W * sizeof(float),
                              cudaMemcpyDeviceToHost));
        snprintf(nombre, sizeof(nombre), "resultados/imagen_%02d_original.png", b);
        guardar_png_rgb(nombre, h_rgb0, H, W);

        CUDA_CHECK(cudaMemcpy(h_gris0, d_gris + (size_t)b * H * W,
                              (size_t)H * W * sizeof(float),
                              cudaMemcpyDeviceToHost));
        snprintf(nombre, sizeof(nombre), "resultados/imagen_%02d_grises.png", b);
        guardar_png_gris(nombre, h_gris0, H, W);

        CUDA_CHECK(cudaMemcpy(h_bordes0, d_bordes + (size_t)b * H * W,
                              (size_t)H * W * sizeof(float),
                              cudaMemcpyDeviceToHost));
        snprintf(nombre, sizeof(nombre), "resultados/imagen_%02d_bordes.png", b);
        guardar_png_gris(nombre, h_bordes0, H, W);

        CUDA_CHECK(cudaMemcpy(h_norm0, d_norm + (size_t)b * H * W,
                              (size_t)H * W * sizeof(float),
                              cudaMemcpyDeviceToHost));
        snprintf(nombre, sizeof(nombre), "resultados/imagen_%02d_normalizada.png", b);
        guardar_png_gris(nombre, h_norm0, H, W);
    }

    /* Archivo de texto con RMSE */
    FILE *fout = fopen("resultados/rmse_por_imagen.txt", "w");
    if (fout) {
        fprintf(fout, "# RMSE por imagen\n");
        fprintf(fout, "# Referencia: imagen 0 convertida a escala de grises\n");
        fprintf(fout, "#\n");
        fprintf(fout, "%-8s %12s\n", "# Imagen", "RMSE");
        for (int b = 0; b < B; b++)
            fprintf(fout, "%-8d %12.6f\n", b, h_rmse[b]);
        fclose(fout);
    }
    printf("PNGs y RMSE guardados en resultados/\n");

    /* ── 11. cudaFree de todos los punteros ───────────────────────────── */
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gris));
    CUDA_CHECK(cudaFree(d_bordes));
    CUDA_CHECK(cudaFree(d_norm));
    CUDA_CHECK(cudaFree(d_maximos));
    CUDA_CHECK(cudaFree(d_rmse));
    CUDA_CHECK(cudaFree(d_ref));

    /* ── 12. Liberar memoria CPU ──────────────────────────────────────── */
    free(h_rgb);
    free(h_rmse);
    free(h_rgb0);
    free(h_gris0);
    free(h_bordes0);
    free(h_norm0);

    return 0;
}
