#include "imagen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── stb_image (single-header library) ──────────────────────────────────────
   Descargar stb_image.h y stb_image_write.h desde https://github.com/nothings/stb
   y colocarlos en utils/ antes de compilar.                                  */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* ── Listado de directorio: POSIX (Linux/WSL) o Win32 ───────────────────── */
#ifdef _WIN32
  #include <windows.h>
#else
  #include <dirent.h>
#endif

/* Devuelve 1 si la extension del archivo es PNG / JPG / JPEG / BMP */
static int es_imagen(const char *nombre) {
    const char *ext = strrchr(nombre, '.');
    if (!ext) return 0;
    return (strcmp(ext, ".png")  == 0 ||
            strcmp(ext, ".jpg")  == 0 ||
            strcmp(ext, ".jpeg") == 0 ||
            strcmp(ext, ".bmp")  == 0);
}

static void agregar_ruta(char ***lista, int *n, const char *ruta) {
    *lista = (char**)realloc(*lista, (size_t)(*n + 1) * sizeof(char*));
    (*lista)[*n] = (char*)malloc(strlen(ruta) + 1);
    strcpy((*lista)[*n], ruta);
    (*n)++;
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* ─────────────────────────────────────────────────────────────────────────── */
float* cargar_imagenes(const char *carpeta, int *B, int *H, int *W) {
    char **archivos = NULL;
    int num = 0;
    char ruta[1024];

#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    char patron[512];
    snprintf(patron, sizeof(patron), "%s\\*.*", carpeta);
    HANDLE hFind = FindFirstFileA(patron, &fd);
    if (hFind == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error: no se puede abrir el directorio '%s'\n", carpeta);
        exit(EXIT_FAILURE);
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && es_imagen(fd.cFileName)) {
            snprintf(ruta, sizeof(ruta), "%s\\%s", carpeta, fd.cFileName);
            agregar_ruta(&archivos, &num, ruta);
        }
    } while (FindNextFileA(hFind, &fd));
    FindClose(hFind);
#else
    DIR *dir = opendir(carpeta);
    if (!dir) {
        fprintf(stderr, "Error: no se puede abrir el directorio '%s'\n", carpeta);
        exit(EXIT_FAILURE);
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (es_imagen(entry->d_name)) {
            snprintf(ruta, sizeof(ruta), "%s/%s", carpeta, entry->d_name);
            agregar_ruta(&archivos, &num, ruta);
        }
    }
    closedir(dir);
#endif

    if (num == 0) {
        fprintf(stderr, "Error: no hay imagenes en '%s'\n", carpeta);
        exit(EXIT_FAILURE);
    }

    qsort(archivos, (size_t)num, sizeof(char*), cmp_str);

    /* Dimensiones de referencia (primera imagen) */
    int w0, h0, c0;
    unsigned char *img0 = stbi_load(archivos[0], &w0, &h0, &c0, 3);
    if (!img0) {
        fprintf(stderr, "Error al cargar imagen: %s\n", archivos[0]);
        exit(EXIT_FAILURE);
    }
    stbi_image_free(img0);

    *B = num;
    *H = h0;
    *W = w0;

    /* Buffer principal: layout (B, 3, H, W) */
    float *datos = (float*)malloc((size_t)num * 3 * h0 * w0 * sizeof(float));
    if (!datos) {
        fprintf(stderr, "Error: malloc fallido para buffer de %d imagenes\n", num);
        exit(EXIT_FAILURE);
    }

    for (int b = 0; b < num; b++) {
        int w, h, c;
        unsigned char *img = stbi_load(archivos[b], &w, &h, &c, 3);
        if (!img) {
            fprintf(stderr, "Error al cargar imagen: %s\n", archivos[b]);
            exit(EXIT_FAILURE);
        }
        if (w != w0 || h != h0) {
            fprintf(stderr, "Error: '%s' tiene tamano distinto (%dx%d vs %dx%d)\n",
                    archivos[b], w, h, w0, h0);
            stbi_image_free(img);
            exit(EXIT_FAILURE);
        }
        /* Reorganizar HWC (stb) -> (3, H, W) para el batch */
        for (int ch = 0; ch < 3; ch++)
            for (int row = 0; row < h0; row++)
                for (int col = 0; col < w0; col++)
                    datos[b * 3 * h0 * w0 + ch * h0 * w0 + row * w0 + col] =
                        img[(row * w0 + col) * 3 + ch] / 255.0f;
        stbi_image_free(img);
    }

    for (int i = 0; i < num; i++) free(archivos[i]);
    free(archivos);
    return datos;
}

/* ─────────────────────────────────────────────────────────────────────────── */
void guardar_png_gris(const char *nombre, float *datos, int H, int W) {
    unsigned char *buf = (unsigned char*)malloc((size_t)H * W);
    for (int i = 0; i < H * W; i++) {
        float v = datos[i] < 0.0f ? 0.0f : (datos[i] > 1.0f ? 1.0f : datos[i]);
        buf[i] = (unsigned char)(v * 255.0f + 0.5f);
    }
    stbi_write_png(nombre, W, H, 1, buf, W);
    free(buf);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Toma imagen b=0 del batch con layout (3, H, W) y la guarda como PNG RGB */
void guardar_png_rgb(const char *nombre, float *datos, int H, int W) {
    unsigned char *buf = (unsigned char*)malloc((size_t)H * W * 3);
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            for (int ch = 0; ch < 3; ch++) {
                float v = datos[ch * H * W + row * W + col];
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                buf[(row * W + col) * 3 + ch] = (unsigned char)(v * 255.0f + 0.5f);
            }
        }
    }
    stbi_write_png(nombre, W, H, 3, buf, W * 3);
    free(buf);
}

/* ─────────────────────────────────────────────────────────────────────────── */
float* cargar_referencia(const char *ruta, int H, int W) {
    int w, h, c;
    unsigned char *img = stbi_load(ruta, &w, &h, &c, 1);  /* forzar grises */
    if (!img || w != W || h != H) {
        if (img) stbi_image_free(img);
        return NULL;
    }
    float *datos = (float*)malloc((size_t)H * W * sizeof(float));
    for (int i = 0; i < H * W; i++)
        datos[i] = img[i] / 255.0f;
    stbi_image_free(img);
    return datos;
}
