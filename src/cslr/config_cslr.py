"""
Configuración del Pipeline CSLR
================================
Constantes, hiperparámetros y rutas para el sistema de
Reconocimiento Continuo de Lenguaje de Señas (CSLR).
"""

import os

# ============================================================
# RUTAS
# ============================================================
_RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DIR_VIDEOS = os.path.join(_RAIZ_PROYECTO, 'videos')
DIR_LANDMARKS_CSLR = os.path.join(_RAIZ_PROYECTO, 'landmarks_cslr')
DIR_MODELOS = os.path.join(_RAIZ_PROYECTO, 'models')

RUTA_MODELO_CSLR = os.path.join(DIR_MODELOS, 'cslr_model.keras')
RUTA_MEJOR_MODELO_CSLR = os.path.join(DIR_MODELOS, 'best_cslr_model.keras')
RUTA_MODELO_CSLR_TFLITE = os.path.join(DIR_MODELOS, 'cslr_model_int8.tflite')
RUTA_LABELS_CSLR = os.path.join(DIR_MODELOS, 'labels_cslr.txt')

# Crear directorio de modelos si no existe
if not os.path.exists(DIR_MODELOS):
    os.makedirs(DIR_MODELOS)

# ============================================================
# LANDMARKS DE MEDIAPIPE
# ============================================================

# Pose Landmarker (sin face mesh)
NUM_LANDMARKS_POSE = 33       # MediaPipe Pose devuelve 33 puntos
COMPLEJIDAD_POSE = 1          # 0=lite, 1=full, 2=heavy
CONFIANZA_DETECCION_POSE = 0.5
CONFIANZA_RASTREO_POSE = 0.5

# Hand Landmarker
NUM_LANDMARKS_MANO = 21       # 21 puntos por mano
MAX_MANOS = 2                 # Máximo 2 manos simultáneas
COMPLEJIDAD_MANOS = 0         # 0=lite (rápido para móvil)
CONFIANZA_DETECCION_MANOS = 0.5
CONFIANZA_RASTREO_MANOS = 0.5

# Coordenadas por landmark
NUM_COORDS = 3                # x, y, z

# ============================================================
# ÍNDICES ANATÓMICOS DE REFERENCIA
# ============================================================

# Pose - Índices de referencia para normalización
IDX_HOMBRO_IZQUIERDO = 11
IDX_HOMBRO_DERECHO = 12
IDX_CADERA_IZQUIERDA = 23
IDX_CADERA_DERECHA = 24

# Mano - Índice de la muñeca (origen de coordenadas locales)
IDX_MUNECA = 0

# Mano - Índices de las yemas de los dedos
INDICES_YEMAS = [4, 8, 12, 16, 20]  # Pulgar, Índice, Medio, Anular, Meñique

# Mano - Articulaciones para cálculo de ángulos de falanges
# Cada tupla: (padre, articulación, hijo) → ángulo en la articulación
ARTICULACIONES_ANGULOS = [
    # Pulgar
    (0, 1, 2),    # CMC
    (1, 2, 3),    # MCP
    (2, 3, 4),    # IP
    # Índice
    (0, 5, 6),    # MCP
    (5, 6, 7),    # PIP
    (6, 7, 8),    # DIP
    # Medio
    (0, 9, 10),   # MCP
    (9, 10, 11),  # PIP
    (10, 11, 12), # DIP
    # Anular
    (0, 13, 14),  # MCP
    (13, 14, 15), # PIP
    (14, 15, 16), # DIP
    # Meñique
    (0, 17, 18),  # MCP
    (17, 18, 19), # PIP
    (18, 19, 20), # DIP
]

# ============================================================
# DIMENSIONES DEL TENSOR DE FEATURES
# ============================================================

# Dimensión de cada componente del feature vector por frame
DIM_POSE = NUM_LANDMARKS_POSE * NUM_COORDS                 # 33 × 3 = 99
DIM_MANO_COORDS = NUM_LANDMARKS_MANO * NUM_COORDS          # 21 × 3 = 63
DIM_DISTANCIAS_YEMAS = 10    # C(5,2) = 10 pares de yemas por mano
DIM_ANGULOS_FALANGES = 15    # 5 dedos × 3 ángulos por mano

# Total de features de una mano (coords + distancias + ángulos)
DIM_MANO_TOTAL = DIM_MANO_COORDS + DIM_DISTANCIAS_YEMAS + DIM_ANGULOS_FALANGES  # 88

# Dimensión total del tensor por frame
DIM_FEATURE_FRAME = DIM_POSE + DIM_MANO_TOTAL * MAX_MANOS  # 99 + 88×2 = 275

# Dimensiones por flujo del encoder
DIM_FLUJO_POSE = DIM_POSE                                   # 99
DIM_FLUJO_MANOS = DIM_MANO_TOTAL * MAX_MANOS                # 176
DIM_FLUJO_CINETICO = DIM_FEATURE_FRAME                      # 275 (delta temporal)

# ============================================================
# PARÁMETROS TEMPORALES
# ============================================================

VENTANA_CNN = 20              # Frames de contexto para el encoder 1D-CNN
TAMANO_BUFFER_FIFO = 90       # Frames en el buffer circular de inferencia
PASO_DESLIZAMIENTO = 10       # Cada cuántos frames se ejecuta predicción
LONGITUD_SECUENCIA_CSLR = 60  # Frames por secuencia de entrenamiento
PASO_VENTANA_CSLR = 15        # Stride para generar secuencias de entrenamiento

# ============================================================
# SELECCIÓN ESTOCÁSTICA DE FRAMES
# ============================================================

UMBRAL_ENERGIA_CINETICA = 0.0005  # Energía mínima para considerar un frame "dinámico"
PROB_ELIMINACION_ESTATICOS = 0.5  # Probabilidad de eliminar un frame estático

# ============================================================
# HIPERPARÁMETROS DEL MODELO
# ============================================================

# Codificador 1D-CNN
FILTROS_CNN_1 = 64
FILTROS_CNN_2 = 128
KERNEL_CNN = 3
DIM_PROYECCION = 128          # Dimensión de proyección antes del Transformer

# Decodificador Transformer
NUM_CABEZAS_ATENCION = 4
NUM_CAPAS_TRANSFORMER = 2
DIM_FFN = 256                 # Dimensión de la red feed-forward interna
DROPOUT_TRANSFORMER = 0.1

# Entrenamiento
TASA_APRENDIZAJE_CSLR = 0.001
EPOCAS_CSLR = 80
TAMANO_LOTE_CSLR = 16
LABEL_SMOOTHING = 0.1

# ============================================================
# CUANTIZACIÓN
# ============================================================

TIPO_CUANTIZACION = 'INT8'    # Cuantización dinámica de enteros de 8 bits
