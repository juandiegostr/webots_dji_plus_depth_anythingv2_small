# TFG Webots - Control de Dron con Estimación de Profundidad usando IA

## Descripción del Proyecto

Este proyecto es una simulación en Webots de un dron DJI Mavic 2 Pro que incorpora control manual mediante teclado y estimación de profundidad en tiempo real utilizando un modelo de inteligencia artificial basado en ONNX (Depth Anything V2). El sistema permite controlar el dron, visualizar mapas de profundidad y grabar vídeo de la cámara del dron.

## Estructura del Proyecto

```
TFG_Webots/
├── controllers/
│   └── controlador_depth_onnx/
│       ├── controlador_depth_onnx.py    # Script principal del controlador
│       ├── depth_anything_vits14_364.onnx  # Modelo ONNX para estimación de profundidad
│       └── runtime.ini                  # Configuración del entorno Python
└── worlds/
    ├── mavic_2_pro.wbt                  # Mundo con entorno rural y árboles
    └── sin_arboles.wbt                  # Mundo sin árboles
```

## Características Principales

- **Control Manual**: Control del dron mediante teclado con movimientos básicos (adelante, atrás, giro, etc.)
- **Estimación de Profundidad**: Uso de IA para generar mapas de profundidad en tiempo real desde la cámara del dron
- **Visualización**: Muestra la imagen de la cámara y el mapa de profundidad coloreado
- **Grabación de Vídeo**: Opción para grabar el vídeo de la cámara en formato MP4
- **Estabilización**: Control PID para mantener la altitud y estabilidad del dron
- **Soporte GPU/CPU**: El modelo ONNX puede ejecutarse en GPU (CUDA) o CPU según disponibilidad

## Requisitos del Sistema

### Software
- **Webots R2025a** o superior
- **Python 3.13** (o compatible)
- **Entorno Conda** con las siguientes librerías:
  - opencv-python
  - numpy
  - onnxruntime-gpu (para GPU) o onnxruntime (para CPU)
  - nvidia-cuda-runtime (si se usa GPU)

### Hardware
- **GPU NVIDIA** recomendada para ejecución del modelo de IA (opcional, funciona en CPU)
- **RAM**: Mínimo 8GB recomendado
- **Espacio en disco**: ~500MB para el modelo ONNX y dependencias

## Instalación y Configuración

### 1. Instalar Webots
Descarga e instala Webots desde el sitio oficial: https://cyberbotics.com/

### 2. Configurar el Entorno Python
El proyecto utiliza un entorno Conda específico. Asegúrate de que el path en `runtime.ini` apunte a tu entorno Python:

```ini
[python]
COMMAND = /ruta/a/tu/entorno/bin/python
```

Instala las dependencias necesarias:
```bash
pip install opencv-python numpy onnxruntime-gpu
```

Para soporte GPU, instala los drivers de NVIDIA y CUDA toolkit correspondiente.

### 3. Verificar el Modelo ONNX
El modelo `depth_anything_vits14_364.onnx` debe estar en la carpeta `controllers/controlador_depth_onnx/`. Si no está presente, descárgalo desde el repositorio oficial de Depth Anything.

## Uso

### Ejecutar la Simulación

1. Abre Webots
2. Carga uno de los mundos:
   - `worlds/mavic_2_pro.wbt` para el entorno con árboles
   - `worlds/sin_arboles.wbt` para el entorno sin árboles
3. El controlador se cargará automáticamente
4. Ejecuta la simulación (botón Play)

### Controles del Teclado

- **↑ (arriba)**: Mover adelante
- **↓ (abajo)**: Mover atrás
- **→ (derecha)**: Girar a la derecha
- **← (izquierda)**: Girar a la izquierda
- **Shift + ↑**: Aumentar altitud objetivo
- **Shift + ↓**: Disminuir altitud objetivo
- **Shift + →**: Desplazamiento lateral derecho
- **Shift + ←**: Desplazamiento lateral izquierdo
- **Q**: Salir de la simulación

### Visualización

Durante la ejecución, se mostrarán dos ventanas:
- **Dron**: Imagen RGB de la cámara del dron
- **IA Depth**: Mapa de profundidad coloreado con información de altitud y distancia central

### Grabación de Vídeo

Por defecto, el sistema graba vídeo de la cámara en `drone_camera.mp4`. Para desactivar:
- Edita la variable `GRABAR_VIDEO = False` en el script

## Parámetros Configurables

### Control del Dron
```python
k_vertical_thrust = 68.5    # Empuje vertical base
k_vertical_offset = 0.6     # Offset vertical
k_vertical_p = 3.0          # Ganancia proporcional vertical
k_roll_p = 50.0             # Ganancia proporcional roll
k_pitch_p = 30.0            # Ganancia proporcional pitch
target_altitude = 1.0       # Altitud objetivo inicial [m]
```

### IA y Procesamiento
```python
INPUT_SIZE = 364             # Tamaño de entrada del modelo
PASOS_A_SALTAR = 8           # Frecuencia de inferencia (cada 8 pasos)
FACTOR_METRICO = 12.0        # Factor de conversión a metros, es experimental y puede que no se ajuste correctamente, dado que depth anything no es un modelo metrico sino relativo
```

### Grabación
```python
GRABAR_VIDEO = True          # Activar/desactivar grabación
VIDEO_FPS = 20.0             # FPS del vídeo
VIDEO_PATH = "drone_camera.mp4"  # Ruta del archivo de salida
```

## Arquitectura Técnica

### Controlador Principal
El script `controlador_depth_onnx.py` maneja:
1. **Inicialización**: Configuración de dispositivos Webots (cámara, IMU, GPS, motores)
2. **Control PID**: Estabilización del dron usando sensores inerciales
3. **Entrada de Usuario**: Procesamiento de comandos del teclado
4. **Inferencia IA**: Estimación de profundidad cada N pasos de simulación
5. **Visualización**: Mostrar imágenes y mapas de profundidad usando OpenCV

### Modelo de IA
- **Depth Anything V2**: Modelo de visión por computadora para estimación monocular de profundidad
- **Formato**: ONNX para compatibilidad multiplataforma
- **Optimización**: Soporte para ejecución en GPU NVIDIA vía CUDA

### Simulación Webots
- **Robot**: DJI Mavic 2 Pro con física realista
- **Sensores**: Cámara RGB, IMU, GPS, giroscopio
- **Actuadores**: 4 motores de hélices con control de velocidad

## Solución de Problemas

### Error de GPU
Si hay problemas con CUDA:
1. Verifica que los drivers NVIDIA estén instalados
2. Instala `onnxruntime` en lugar de `onnxruntime-gpu`
3. El sistema automáticamente caerá en CPU

### Error de Librerías
Asegúrate de que todas las librerías estén instaladas en el entorno especificado en `runtime.ini`

### Rendimiento
- Reduce `VIDEO_FPS` si la simulación es lenta
- Aumenta `PASOS_A_SALTAR` para reducir frecuencia de inferencia IA
- Usa CPU si GPU no está disponible

## Desarrollo y Contribución

Este proyecto forma parte de un Trabajo de Fin de Grado (TFG) en Ingeniería Robótica. 

## Licencias y Créditos

- **Webots**: LGPL 2.1
- **Depth Anything**: MIT License (Lihe Yang et al.)
- **OpenCV, NumPy, ONNX Runtime**: Licencias respectivas
