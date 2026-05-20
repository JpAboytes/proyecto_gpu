#ifndef IMAGEN_H
#define IMAGEN_H

/* Carga todas las imagenes PNG/JPG de una carpeta a un buffer float [0,1].
   Layout de salida: (B, 3, H, W) row-major.
   Devuelve buffer alocado con malloc; el llamador libera con free(). */
float* cargar_imagenes(const char *carpeta, int *B, int *H, int *W);

/* Guarda un arreglo float [0,1] de tamano H*W como PNG en escala de grises */
void guardar_png_gris(const char *nombre, float *datos, int H, int W);

/* Guarda la imagen b=0 de un batch (layout 3,H,W) como PNG RGB */
void guardar_png_rgb(const char *nombre, float *datos, int H, int W);

/* Carga una sola imagen como referencia en escala de grises (H x W).
   Devuelve NULL si la ruta no existe o las dimensiones no coinciden. */
float* cargar_referencia(const char *ruta, int H, int W);

#endif /* IMAGEN_H */
