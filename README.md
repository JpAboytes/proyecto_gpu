# proyecto_gpu — Pipeline de Procesamiento de Imágenes en CUDA

## Descripción

Pipeline GPU completo que procesa un batch de imágenes RGB a través de cuatro kernels CUDA encadenados. Los datos permanecen en GPU durante todo el pipeline; solo se transfieren al inicio (H→D) y al final (D→H) para guardar resultados.

```
imagenes/ (RGB)
    │
    ▼  cudaMemcpy H→D
    │
    ▼  K1 — Escala de grises   (B,3,H,W) → (B,H,W)
    │
    ▼  K2 — Bordes Sobel       (B,H,W)   → (B,H,W)
    │
    ▼  K3 — Normalizar         (B,H,W)   → (B,H,W)  + reduce max
    │
    ▼  K4 — RMSE vs ref        (B,H,W)   → (B,)
    │
    ▼  cudaMemcpy D→H
    │
resultados/ (PNGs + rmse_por_imagen.txt)
```

---

## Estructura del proyecto

```
proyecto_gpu/
├── main.cu
├── kernels/
│   ├── grises.cu / grises.h        ← K1: luminancia BT.601
│   ├── bordes.cu / bordes.h        ← K2: filtro Sobel
│   ├── normalizar.cu / normalizar.h ← K3: reducción max + división
│   └── mse.cu / mse.h              ← K4: RMSE por imagen
├── utils/
│   ├── imagen.cu / imagen.h        ← I/O PNG con stb_image
│   └── timer.cu / timer.h          ← cronómetro cudaEvent
├── imagenes/                       ← colocar imágenes aquí
├── resultados/                     ← PNGs generados
└── README.md
```

---

## Dependencias

### CUDA Toolkit ≥ 11.0

Verifica que `nvcc` esté disponible en el PATH:

```bash
nvcc --version
```

### stb_image (single-header library)

Descarga los siguientes archivos desde **https://github.com/nothings/stb** y colócalos en `utils/`:

```
utils/stb_image.h
utils/stb_image_write.h
```

Descarga directa (Linux/WSL):

```bash
wget -P utils/ https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget -P utils/ https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

---

## Compilación

```bash
nvcc -O2 -o pipeline main.cu kernels/grises.cu kernels/bordes.cu \
     kernels/normalizar.cu kernels/mse.cu utils/imagen.cu utils/timer.cu -lm
```

> **Windows (NVCC + MSVC)**: ejecutar desde el *Developer Command Prompt* con el CUDA Toolkit instalado.  
> **Linux / WSL**: el comando anterior funciona directamente desde la raíz del proyecto.

---

## Ejecución

1. Copia al menos una imagen PNG o JPG en la carpeta `imagenes/`.
2. Ejecuta desde la raíz del proyecto:

```bash
./pipeline          # Linux / WSL
pipeline.exe        # Windows
```

Salida esperada en consola:

```
Imagenes cargadas: 3  |  Tamano: 512 x 512

[H->D RGB]        0.8241 ms
[K1 Grises]       0.0512 ms
[K2 Bordes]       0.0634 ms
[K3 Normalizar]   0.0318 ms
[K4 MSE]          0.0201 ms
[D->H RMSE]       0.0089 ms

=========================================
Operacion                 Tiempo (ms)
-----------------------------------------
Transferencia H->D          0.8241
K1 Grises                   0.0512
K2 Bordes (Sobel)           0.0634
K3 Normalizar               0.0318
K4 MSE                      0.0201
Transferencia D->H          0.0089
=========================================

RMSE por imagen (referencia = imagen 0 en grises):
  [  0]  0.000000
  [  1]  0.134251
  [  2]  0.198743
```

---

## Tabla de tiempos (completar con resultados reales)

| Operación             | Tiempo (ms) |
|-----------------------|-------------|
| Transferencia H→D     |             |
| K1 Grises             |             |
| K2 Bordes (Sobel)     |             |
| K3 Normalizar         |             |
| K4 MSE                |             |
| Transferencia D→H     |             |
| **Total**             |             |

---

## Descripción de cada kernel

### K1 — `escala_grises` (`kernels/grises.cu`)

- **Grid**: `dim3(⌈W/16⌉, ⌈H/16⌉)` · **Bloque**: `dim3(16, 16)` = 256 hilos (8 warps)
- Cada hilo procesa el píxel `(fila, col)` de todas las imágenes del batch en un loop interno.
- Fórmula: `gris = 0.2989·R + 0.5870·G + 0.1140·B` (pesos ITU-R BT.601)
- Layout entrada `(B, 3, H, W)`, layout salida `(B, H, W)`.

### K2 — `filtro_sobel` (`kernels/bordes.cu`)

- **Grid**: `dim3(⌈W/16⌉, ⌈H/16⌉)` · **Bloque**: `dim3(16, 16)`
- Aplica convolución 3×3 con `Gx = [[-1,0,+1],[-2,0,+2],[-1,0,+1]]` y `Gy = [[-1,-2,-1],[0,0,0],[+1,+2,+1]]`.
- Magnitud: `sqrt(Gx² + Gy²)`. Píxeles en el borde de la imagen → `0.0f`.

### K3 — Dos kernels en secuencia (`kernels/normalizar.cu`)

| Kernel              | Grid                    | Descripción |
|---------------------|-------------------------|-------------|
| `kernel_reduce_max` | `dim3(B)` × 256 hilos   | Reducción paralela en árbol con `__shared__`; halla el máximo de cada imagen |
| `kernel_dividir`    | `dim3(⌈W/16⌉, ⌈H/16⌉)` × `dim3(16,16)` | Divide cada píxel entre el máximo de su imagen; guarda contra división por cero (`< 1e-8f`) |

### K4 — `kernel_mse` (`kernels/mse.cu`)

- **Grid**: `dim3(B)` · **Bloque**: `dim3(256)` = un bloque por imagen
- Reducción paralela de `Σ(batch[b][i] − ref[i])²` con `__shared__ float sdata[256]`.
- Salida: `RMSE[b] = sqrt(suma / (H·W))`.

---

## Resultados

Después de la ejecución, `resultados/` contendrá:

| Archivo                       | Descripción                               |
|-------------------------------|-------------------------------------------|
| `imagen_00_original.png`      | Primera imagen del batch (RGB)            |
| `imagen_00_grises.png`        | Salida de K1 (luminancia)                 |
| `imagen_00_bordes.png`        | Salida de K2 (magnitud Sobel)             |
| `imagen_00_normalizada.png`   | Salida de K3 (bordes normalizados [0,1])  |
| `rmse_por_imagen.txt`         | RMSE de cada imagen respecto a imagen 0   |

### Verificación del pipeline

<!-- Insertar imagen verificacion_pipeline.png aquí -->

![verificacion_pipeline](resultados/verificacion_pipeline.png)
