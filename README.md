# Pipeline de Procesamiento de Imágenes por GPU en CUDA

**Proyecto Final — Programación Avanzada por GPU**

---

## Descripción general

Este proyecto implementa un **pipeline de visión computacional completamente residente en GPU**, escrito en CUDA C++. El sistema recibe un lote (_batch_) de imágenes RGB, aplica cuatro etapas de transformación mediante _kernels_ CUDA encadenados y produce imágenes procesadas junto con una métrica de similitud (RMSE) para cada imagen del lote.

El principio central de diseño es la **minimización de transferencias PCI-Express**: los datos suben a la GPU una sola vez al inicio (`H→D`), todas las transformaciones intermedias ocurren exclusivamente en memoria de dispositivo y solo los resultados finales regresan a la CPU (`D→H`). Esto elimina las latencias de transferencia que serían el cuello de botella si cada etapa requiriera un viaje de regreso a memoria del host.

---

## Contexto académico — Programación Avanzada por GPU

El proyecto integra y demuestra los conceptos clave de la materia:

| Concepto | Dónde se aplica en el proyecto |
|---|---|
| Jerarquía de memoria GPU | Memoria global (`cudaMalloc`), memoria compartida (`__shared__`) en K3 y K4 |
| Organización de hilos | Grids 2D `(⌈W/16⌉ × ⌈H/16⌉)` para kernels de píxel; Grids 1D `(B)` para reducciones por imagen |
| Patrones de acceso a memoria | Índices `[b·H·W + fila·W + col]` para maximizar coalescencia en filas |
| Reducción paralela en árbol | `kernel_reduce_max` (K3) y `kernel_mse` (K4) con sincronización explícita por `__syncthreads()` |
| Copia dispositivo→dispositivo | La imagen de referencia se copia con `cudaMemcpyDeviceToDevice` sin bajar a CPU |
| Temporización con CUDA Events | `cudaEvent_t` para medir con precisión cada etapa del pipeline |
| Manejo de errores CUDA | Macro `CUDA_CHECK()` en todas las llamadas a la API |
| Procesamiento en batch | Dimensión `B` procesada con un loop interno dentro de cada kernel |

---

## Pipeline de procesamiento

```
imagenes/  (B imágenes JPEG/PNG, resolución idéntica)
    │
    ▼  cudaMemcpy  H→D      (B × 3 × H × W floats)
    │
    ▼  K1 — escala_grises   (B,3,H,W) → (B,H,W)
    │         Luminancia ITU-R BT.601: gris = 0.2989·R + 0.5870·G + 0.1140·B
    │
    ▼  K2 — filtro_sobel    (B,H,W)   → (B,H,W)
    │         Convolución 3×3 Sobel: M = √(Gx² + Gy²)
    │
    ▼  K3 — lanzar_normalizar  (B,H,W) → (B,H,W)
    │         [Paso A] kernel_reduce_max: reducción paralela → máximo por imagen
    │         [Paso B] kernel_dividir:   pixel / max_imagen ∈ [0, 1]
    │
    ▼  K4 — kernel_mse       (B,H,W) + referencia → (B,)
    │         Reducción paralela: RMSE[b] = √( Σ(norm[b][i] − ref[i])² / (H·W) )
    │
    ▼  cudaMemcpy  D→H      (B floats — vector RMSE)
    │
resultados/  (4 PNGs de imagen_00 + rmse_por_imagen.txt)
```

> La referencia para el cálculo de RMSE es la imagen 0 convertida a escala de grises.
> La copia de referencia se realiza con `cudaMemcpyDeviceToDevice`, sin pasar por CPU.

---

## Estructura del proyecto

```
proyecto_gpu/
├── main.cu                          ← Orquestador del pipeline
│
├── kernels/
│   ├── grises.cu  / grises.h        ← K1: conversión a escala de grises
│   ├── bordes.cu  / bordes.h        ← K2: detección de bordes Sobel
│   ├── normalizar.cu / normalizar.h ← K3: reducción de máximo + normalización
│   └── mse.cu     / mse.h           ← K4: cálculo de RMSE por imagen
│
├── utils/
│   ├── imagen.cu  / imagen.h        ← Carga/guardado de imágenes con stb_image
│   ├── timer.cu   / timer.h         ← Temporizador con cudaEvent_t
│   ├── stb_image.h                  ← Librería single-header (lectura de imágenes)
│   └── stb_image_write.h            ← Librería single-header (escritura de imágenes)
│
├── imagenes/                        ← Colocar aquí las imágenes de entrada
│   └── 1.jpeg … 6.jpeg
│
├── resultados/                      ← Salidas generadas por el pipeline
│   ├── imagen_00_original.png
│   ├── imagen_00_grises.png
│   ├── imagen_00_bordes.png
│   ├── imagen_00_normalizada.png
│   └── rmse_por_imagen.txt
│
└── README.md
```

---

## Descripción de cada kernel

### K1 — `escala_grises` (`kernels/grises.cu`)

Convierte el batch RGB a escala de grises usando la fórmula de luminancia perceptual **ITU-R BT.601**.

| Parámetro | Valor | Justificación |
|---|---|---|
| Grid | `dim3(⌈W/16⌉, ⌈H/16⌉)` | Cubre todos los píxeles de la imagen |
| Bloque | `dim3(16, 16)` = 256 hilos | 8 warps completos; múltiplo de 32 |
| Memoria compartida | No requerida | Cada hilo lee 3 valores y escribe 1 |
| Loop sobre B | Interno al kernel | Un lanzamiento procesa todo el batch |

**Fórmula:**
```
gris = 0.2989·R + 0.5870·G + 0.1140·B
```
Los coeficientes reflejan la sensibilidad del ojo humano: mayor peso al verde, menor al azul.

**Layout de memoria:** entrada `(B, 3, H, W)` en orden canal-mayor (CHW); salida `(B, H, W)`.

---

### K2 — `filtro_sobel` (`kernels/bordes.cu`)

Aplica el **operador de Sobel** para detectar bordes mediante convolución 3×3 separada en gradiente horizontal y vertical.

| Parámetro | Valor |
|---|---|
| Grid | `dim3(⌈W/16⌉, ⌈H/16⌉)` |
| Bloque | `dim3(16, 16)` = 256 hilos |

**Kernels de convolución:**

```
Gx = [[-1,  0, +1],      Gy = [[-1, -2, -1],
      [-2,  0, +2],            [ 0,  0,  0],
      [-1,  0, +1]]            [+1, +2, +1]]

Magnitud = √(Gx² + Gy²)
```

- Gx detecta transiciones horizontales (bordes verticales en la imagen).
- Gy detecta transiciones verticales (bordes horizontales en la imagen).
- Los píxeles en el borde de la imagen (sin vecindad 3×3 completa) se asignan a `0.0f`.

---

### K3 — Normalización por máximo (`kernels/normalizar.cu`)

Requiere dos kernels en secuencia porque la normalización necesita conocer el máximo global de cada imagen antes de dividir.

#### Paso A — `kernel_reduce_max`

Encuentra el valor máximo de cada imagen mediante **reducción paralela en árbol** con memoria compartida.

| Parámetro | Valor | Justificación |
|---|---|---|
| Grid | `dim3(B)` | Un bloque por imagen |
| Bloque | `dim3(256)` | 8 warps completos |
| Memoria compartida | `256 × sizeof(float)` | Buffer de reducción local al bloque |

**Algoritmo de reducción:**
```
1. Cada hilo recorre sus elementos con stride = 256 y guarda el máximo local en sdata[tid].
2. __syncthreads()
3. Reducción en árbol: stride 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
4. Hilo 0 escribe sdata[0] → d_maximos[b]
```

#### Paso B — `kernel_dividir`

Divide cada píxel entre el máximo de su imagen (calculado en el Paso A).

| Parámetro | Valor |
|---|---|
| Grid | `dim3(⌈W/16⌉, ⌈H/16⌉)` |
| Bloque | `dim3(16, 16)` |

Protección contra división por cero: si `max < 1e-8f`, el valor del píxel se conserva sin modificar.

---

### K4 — `kernel_mse` (`kernels/mse.cu`)

Calcula el **RMSE** (_Root Mean Square Error_) de cada imagen normalizada respecto a la imagen de referencia (imagen 0 en escala de grises).

| Parámetro | Valor | Justificación |
|---|---|---|
| Grid | `dim3(B)` | Un bloque por imagen |
| Bloque | `dim3(256)` | 8 warps completos |
| Memoria compartida | `256 × sizeof(float)` | Buffer de reducción local al bloque |

**Algoritmo:**
```
1. Cada hilo acumula su suma parcial: Σ (batch[b][i] − ref[i])²  (stride = 256)
2. Reducción paralela en árbol en sdata[]
3. Hilo 0 calcula: RMSE[b] = √(sdata[0] / (H·W))
```

---

## Estrategia de memoria y transferencias

```
Host (CPU/RAM)                         Dispositivo (GPU/VRAM)
─────────────                          ──────────────────────
h_rgb[B×3×H×W]  ──── H→D ────►  d_rgb[B×3×H×W]
                                        │  K1
                                  d_gris[B×H×W]
                                        │  K2
                                  d_bordes[B×H×W]
                                        │  K3
                                  d_norm[B×H×W]
                                  d_maximos[B]  (temporal K3)
                                        │  K4
                                  d_rmse[B]
                                        │
h_rmse[B]       ◄─── D→H ────── d_rmse[B]

(Para guardar PNGs: transferencias D→H adicionales solo de imagen_00)
```

Las únicas transferencias PCI-Express del pipeline principal son:
- **Una subida** (`H→D`): el batch RGB completo al inicio.
- **Una bajada** (`D→H`): el vector RMSE (B floats) al final.

Las imágenes intermedias (grises, bordes, normalizada) solo se bajan para la primera imagen con fines de visualización.

---

## Temporización con CUDA Events

Cada etapa del pipeline se mide con `cudaEvent_t` (no con `clock()` del host), lo que garantiza que se mide el tiempo real de ejecución en GPU incluyendo la sincronización de la jerarquía de kernels:

```c
timer_start(&t);          // cudaEventRecord(start)
lanzar_grises(...);
ms_grises = timer_stop(&t); // cudaEventSynchronize(stop) → elapsed ms
```

`cudaDeviceSynchronize()` se llama al final de cada kernel para garantizar mediciones aisladas entre etapas.

---

## Resultados obtenidos

### RMSE por imagen (referencia = imagen 0 normalizada)

| Imagen | RMSE |
|--------|------|
| 0 (referencia) | 0.587404 |
| 1 | 0.603116 |
| 2 | 0.598270 |
| 3 | 0.596376 |
| 4 | 0.583873 |
| 5 | 0.610736 |

> Los valores de RMSE no son cero para la referencia porque se compara la imagen normalizada de bordes contra la imagen en grises sin procesar de bordes (las etapas K2 y K3 se aplican al batch completo pero la referencia es la imagen 0 solo después de K1).

### Tabla de tiempos del pipeline

Medición con imágenes de **1600 × 1200 px**, batch de **6 imágenes**, GPU con CUDA Events.

| Operación | Tiempo (ms) | % del total |
|---|---|---|
| Transferencia H→D | 11.3938 | 51.8 % |
| K1 Grises | 1.8363 | 8.3 % |
| K2 Bordes (Sobel) | 0.9708 | 4.4 % |
| K3 Normalizar | 3.5011 | 15.9 % |
| K4 MSE | 3.7857 | 17.2 % |
| Transferencia D→H | 0.2157 | 1.0 % |
| **Total** | **21.7034** | 100 % |

> La transferencia H→D domina con el 51.8 % del tiempo total, confirmando que el bus PCI-Express es el cuello de botella real, no el cómputo. Los cuatro kernels juntos suman solo 10.1 ms (46.2 %). La descarga D→H es negligible porque solo transfiere 6 floats (el vector RMSE).

### Imágenes de salida (imagen 0)

| Archivo | Descripción |
|---|---|
| `resultados/imagen_00_original.png` | Imagen original en RGB |
| `resultados/imagen_00_grises.png` | Salida de K1 — luminancia perceptual |
| `resultados/imagen_00_bordes.png` | Salida de K2 — magnitud del gradiente Sobel |
| `resultados/imagen_00_normalizada.png` | Salida de K3 — bordes normalizados a [0, 1] |
| `resultados/rmse_por_imagen.txt` | RMSE de todas las imágenes respecto a imagen 0 |

---

## Dependencias

### CUDA Toolkit ≥ 11.0

```powershell
nvcc --version
```

Descarga: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### stb_image (single-header, sin dependencias externas)

Las librerías `stb_image.h` y `stb_image_write.h` ya se incluyen en `utils/`. Si se requiere actualizar:

```bash
# Linux / WSL
wget -P utils/ https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget -P utils/ https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

---

## Compilación

### Windows (Developer Command Prompt con CUDA Toolkit)

```bat
nvcc -O2 -o proyecto_gpu.exe main.cu kernels/grises.cu kernels/bordes.cu ^
     kernels/normalizar.cu kernels/mse.cu utils/imagen.cu utils/timer.cu
```

### Linux / WSL

```bash
nvcc -O2 -o pipeline main.cu kernels/grises.cu kernels/bordes.cu \
     kernels/normalizar.cu kernels/mse.cu utils/imagen.cu utils/timer.cu -lm
```

---

## Ejecución

1. Colocar al menos una imagen PNG, JPG o JPEG en la carpeta `imagenes/`.
   - Todas las imágenes deben tener las **mismas dimensiones**; el pipeline valida esto al cargar.
2. Ejecutar desde la raíz del proyecto:

```bat
proyecto_gpu.exe        # Windows
./pipeline              # Linux / WSL
```

**Salida de consola esperada:**

```
Imagenes cargadas: 6  |  Tamano: 640 x 480

[H->D RGB]        x.xxxx ms
[K1 Grises]       x.xxxx ms
[K2 Bordes]       x.xxxx ms
[K3 Normalizar]   x.xxxx ms
[K4 MSE]          x.xxxx ms
[D->H RMSE]       x.xxxx ms

=========================================
Operacion                    Tiempo (ms)
-----------------------------------------
Transferencia H->D               x.xxxx
K1 Grises                        x.xxxx
K2 Bordes (Sobel)                x.xxxx
K3 Normalizar                    x.xxxx
K4 MSE                           x.xxxx
Transferencia D->H               x.xxxx
=========================================

RMSE por imagen (referencia = imagen 0 en grises):
  [  0]  0.587404
  [  1]  0.603116
  ...

PNGs y RMSE guardados en resultados/
```

---

## Detalles de implementación relevantes

### Acceso coalescente a memoria

Los arrays en GPU usan layout **CHW** (canal-alto-ancho) con indexación `[b·H·W + fila·W + col]`. Con el grid 2D configurado como `(W/16, H/16)`, los hilos adyacentes en x acceden a columnas consecutivas de la misma fila, produciendo accesos coalescentes en la dirección rápida de la memoria global.

### Tamaño de bloque: 256 hilos (8 warps)

- Los bloques de 256 hilos garantizan un número entero de warps (8 × 32 = 256).
- Para las reducciones (K3 y K4) el tamaño de 256 permite 8 iteraciones de reducción en árbol (256 → 1), cubriendo todos los niveles sin desperdicio.
- Para los kernels 2D (K1, K2, K3-B) el bloque `16×16 = 256` es un compromiso estándar entre ocupancia y reuso de caché L1.

### Protección de bordes en Sobel

El kernel Sobel asigna magnitud cero a los píxeles del borde de la imagen (primera y última fila/columna) porque no disponen de los 8 vecinos necesarios para la convolución 3×3. Esto evita lecturas fuera de límites sin recurrir a padding adicional.

### Manejo de errores con `CUDA_CHECK`

Todas las llamadas a la API de CUDA están envueltas en la macro `CUDA_CHECK`, que imprime el archivo, línea y mensaje de error de CUDA antes de terminar el programa con `EXIT_FAILURE`. Esto facilita la depuración en fases de desarrollo.

---

