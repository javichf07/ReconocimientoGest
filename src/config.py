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
