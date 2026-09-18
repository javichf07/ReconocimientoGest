import os

# Dataset
DIR_DATOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
ALTO_IMG = 224
ANCHO_IMG = 224
TAMANO_IMG = (ALTO_IMG, ANCHO_IMG)

# Modelo
TAMANO_LOTE = 32
TASA_APRENDIZAJE = 0.0001
EPOCAS = 20
NUM_CLASES = 0 # Se actualizará dinámicamente basado en el dataset

# Rutas
RUTA_MODELOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
if not os.path.exists(RUTA_MODELOS):
    os.makedirs(RUTA_MODELOS)

RUTA_MEJOR_MODELO = os.path.join(RUTA_MODELOS, 'best_model.keras')
RUTA_MODELO_FINAL = os.path.join(RUTA_MODELOS, 'final_model.keras')
RUTA_MODELO_TFLITE = os.path.join(RUTA_MODELOS, 'model.tflite')
MOBILENET_ALFA = 1.0 # 0.35, 0.50, 0.75, 1.0

# ============================================================
# ENFOQUE C: Landmarks + LSTM (Pipeline principal)
# ============================================================

# Landmarks de MediaPipe Holistic
# Pose: 33 puntos × 3 (x,y,z) = 99 valores
# Mano Izquierda: 21 puntos × 3 = 63 valores
# Mano Derecha: 21 puntos × 3 = 63 valores
# Total por frame: 75 puntos × 3 = 225 valores
NUM_PUNTOS_POSE = 33
NUM_PUNTOS_MANO = 21
NUM_COORDENADAS = 3  # x, y, z
NUM_LANDMARKS = (NUM_PUNTOS_POSE + NUM_PUNTOS_MANO * 2) * NUM_COORDENADAS  # 225

# Secuencias temporales
LONGITUD_SECUENCIA = 30  # Frames por secuencia (~1 segundo a 30fps)
PASO_VENTANA = 15  # Stride del sliding window (50% overlap)

# Datos de landmarks
DIR_LANDMARKS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'landmarks_data')

# Modelo LSTM
RUTA_MEJOR_MODELO_LSTM = os.path.join(RUTA_MODELOS, 'best_lstm_model.keras')
RUTA_MODELO_FINAL_LSTM = os.path.join(RUTA_MODELOS, 'final_lstm_model.keras')
RUTA_MODELO_LSTM_TFLITE = os.path.join(RUTA_MODELOS, 'lstm_model.tflite')

# Hiperparámetros LSTM
UNIDADES_LSTM_1 = 128
UNIDADES_LSTM_2 = 64
UNIDADES_DENSA = 64
DROPOUT = 0.3
TASA_APRENDIZAJE_LSTM = 0.001
EPOCAS_LSTM = 50
TAMANO_LOTE_LSTM = 32
