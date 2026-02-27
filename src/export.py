import tensorflow as tf
import os
from . import config

def convertir_a_tflite():
    print(f"Cargando modelo desde {config.RUTA_MEJOR_MODELO}...")
    try:
        modelo = tf.keras.models.load_model(config.RUTA_MEJOR_MODELO)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    print("Convirtiendo a TFLite...")
    convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    
    # Optimización: Optimizaciones por defecto (incluye cuantización)
    convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Opcional: Cuantización Float16 para compatibilidad con GPU delegate o tamaño más pequeño
    # convertidor.target_spec.supported_types = [tf.float16]

    modelo_tflite = convertidor.convert()

    print(f"Guardando modelo TFLite en {config.RUTA_MODELO_TFLITE}...")
    with open(config.RUTA_MODELO_TFLITE, 'wb') as f:
        f.write(modelo_tflite)
        
    tamano_en_mb = os.path.getsize(config.RUTA_MODELO_TFLITE) / (1024 * 1024)
    print(f"Tamaño Modelo TFLite: {tamano_en_mb:.2f} MB")
    
    if tamano_en_mb > 20:
        print("ADVERTENCIA: El tamaño del modelo excede el límite de 20MB.")
    else:
        print("El tamaño del modelo está dentro del límite.")

if __name__ == "__main__":
    convertir_a_tflite()
