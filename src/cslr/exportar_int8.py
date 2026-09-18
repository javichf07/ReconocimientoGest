"""
Módulo 5: Exportación con Cuantización Int8
=============================================
Convierte el modelo CSLR entrenado a TensorFlow Lite
aplicando Cuantización Dinámica de Enteros de 8 bits (Int8)
para reducir el tamaño y prepararlo para aceleradores NPU/DSP.

Uso:
    python -m src.cslr.exportar_int8
"""

import tensorflow as tf
import numpy as np
import os
from . import config_cslr as cfg
from . import dataset_cslr


def generar_dataset_representativo(X_data, num_muestras=100):
    """
    Genera un dataset representativo para calibración de la cuantización Int8.
    
    La cuantización Int8 necesita datos representativos para determinar
    el rango óptimo de valores enteros que representen los pesos float.
    
    Args:
        X_data: np.array (N, T, 275) datos de entrenamiento
        num_muestras: int, número de muestras para calibración
        
    Yields:
        lista de np.arrays con las entradas del modelo
    """
    indices = np.random.choice(len(X_data), min(num_muestras, len(X_data)), replace=False)
    
    for idx in indices:
        muestra = X_data[idx:idx+1]  # (1, T, 275)
        
        # Separar en los 3 flujos
        pose = muestra[:, :, :cfg.DIM_FLUJO_POSE].astype(np.float32)
        manos = muestra[:, :, cfg.DIM_FLUJO_POSE:].astype(np.float32)
        
        cinetico = np.zeros_like(muestra, dtype=np.float32)
        cinetico[:, 1:, :] = muestra[:, 1:, :] - muestra[:, :-1, :]
        
        yield [pose, manos, cinetico]


def exportar_modelo_int8():
    """
    Exporta el modelo CSLR a TFLite con cuantización Int8.
    
    Pipeline:
    1. Carga el mejor modelo .keras
    2. Configura el conversor con INT8 quantization
    3. Usa datos representativos para calibración
    4. Guarda el modelo cuantizado
    """
    print("=" * 60)
    print("EXPORTACIÓN CSLR - CUANTIZACIÓN INT8")
    print("=" * 60)
    
    # 1. Cargar modelo (reconstruir arquitectura + cargar pesos)
    if not os.path.exists(cfg.RUTA_MEJOR_MODELO_CSLR):
        print(f"Modelo no encontrado: {cfg.RUTA_MEJOR_MODELO_CSLR}")
        print("Ejecuta el entrenamiento primero.")
        return
    
    # Leer número de clases desde labels
    if not os.path.exists(cfg.RUTA_LABELS_CSLR):
        print(f"Labels no encontradas: {cfg.RUTA_LABELS_CSLR}")
        return
    
    with open(cfg.RUTA_LABELS_CSLR, 'r', encoding='utf-8') as f:
        nombres_clases = [l.strip() for l in f.readlines() if l.strip()]
    num_clases = len(nombres_clases)
    print(f"Clases ({num_clases}): {nombres_clases}")
    
    # Reconstruir la arquitectura del modelo desde código
    # (evita problemas de deserialización de capas custom como CodificadorCNN1D,
    #  AtencionEuclidiana, DecodificadorTransformer)
    print(f"Reconstruyendo arquitectura del modelo...")
    from .modelo.modelo_cslr import construir_modelo_cslr
    modelo = construir_modelo_cslr(
        num_clases=num_clases,
        longitud_secuencia=cfg.LONGITUD_SECUENCIA_CSLR
    )
    
    # Cargar pesos desde el archivo .keras
    print(f"Cargando pesos desde: {cfg.RUTA_MEJOR_MODELO_CSLR}")
    try:
        # Intentar cargar pesos directamente (funciona en TF 2.15)
        modelo.load_weights(cfg.RUTA_MEJOR_MODELO_CSLR)
    except Exception as e1:
        print(f"Carga directa falló ({e1}), intentando extraer pesos del .keras...")
        try:
            # Fallback: extraer model.weights.h5 del archivo .keras (es un zip)
            import zipfile
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(cfg.RUTA_MEJOR_MODELO_CSLR, 'r') as z:
                    z.extractall(tmpdir)
                ruta_pesos = os.path.join(tmpdir, 'model.weights.h5')
                if os.path.exists(ruta_pesos):
                    modelo.load_weights(ruta_pesos)
                    print("Pesos cargados exitosamente desde archivo .keras extraído.")
                else:
                    print(f"No se encontró model.weights.h5 dentro del .keras")
                    print(f"Contenido del .keras: {os.listdir(tmpdir)}")
                    return
        except Exception as e2:
            print(f"Error cargando pesos: {e2}")
            print("Intenta re-entrenar el modelo: python -m src.cslr.entrenamiento")
            return
    
    # 2. Cargar datos para calibración
    print("Cargando datos representativos para calibración...")
    resultado = dataset_cslr.cargar_datos()
    
    if resultado[0] is not None:
        X_train = resultado[0]
        tiene_datos_calibracion = True
    else:
        print("⚠ Sin datos de calibración. Usando cuantización sin calibración.")
        tiene_datos_calibracion = False
    
    # 3. Configurar conversor TFLite
    print("\nConfigurando conversor TFLite...")
    convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    
    # Cuantización Int8 Dinámica
    convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Soporte para operaciones de Transformer/LSTM en TFLite
    convertidor.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    convertidor._experimental_lower_tensor_list_ops = False
    
    # Int8 con datos representativos para calibración completa
    if tiene_datos_calibracion:
        convertidor.representative_dataset = lambda: generar_dataset_representativo(X_train)
        
        # Forzar cuantización Int8 de entrada/salida también
        convertidor.target_spec.supported_types = [tf.int8]
        convertidor.inference_input_type = tf.float32   # Mantener float en I/O para compatibilidad
        convertidor.inference_output_type = tf.float32
    
    # 4. Convertir
    print("Convirtiendo modelo a TFLite Int8...")
    try:
        modelo_tflite = convertidor.convert()
    except Exception as e:
        print(f"\nError en conversión Int8 completa: {e}")
        print("Intentando cuantización dinámica sin calibración...")
        
        # Fallback: cuantización dinámica simple
        convertidor2 = tf.lite.TFLiteConverter.from_keras_model(modelo)
        convertidor2.optimizations = [tf.lite.Optimize.DEFAULT]
        convertidor2.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        convertidor2._experimental_lower_tensor_list_ops = False
        modelo_tflite = convertidor2.convert()
    
    # 5. Guardar
    with open(cfg.RUTA_MODELO_CSLR_TFLITE, 'wb') as f:
        f.write(modelo_tflite)
    
    tamano_mb = os.path.getsize(cfg.RUTA_MODELO_CSLR_TFLITE) / (1024 * 1024)
    
    # Tamaño del modelo original para comparar
    tamano_original = 0
    if os.path.exists(cfg.RUTA_MEJOR_MODELO_CSLR):
        # El .keras es un directorio a veces, medir tamaño total
        if os.path.isdir(cfg.RUTA_MEJOR_MODELO_CSLR):
            for root, dirs, files in os.walk(cfg.RUTA_MEJOR_MODELO_CSLR):
                for file in files:
                    tamano_original += os.path.getsize(os.path.join(root, file))
        else:
            tamano_original = os.path.getsize(cfg.RUTA_MEJOR_MODELO_CSLR)
        tamano_original_mb = tamano_original / (1024 * 1024)
    else:
        tamano_original_mb = 0
    
    print(f"\n{'='*60}")
    print(f"EXPORTACIÓN COMPLETADA")
    print(f"{'='*60}")
    print(f"Modelo TFLite: {cfg.RUTA_MODELO_CSLR_TFLITE}")
    print(f"Tamaño TFLite Int8: {tamano_mb:.2f} MB")
    if tamano_original_mb > 0:
        reduccion = (1 - tamano_mb / tamano_original_mb) * 100
        print(f"Tamaño original: {tamano_original_mb:.2f} MB")
        print(f"Reducción: {reduccion:.0f}%")
    
    if tamano_mb < 5:
        print("✓ Ultra-ligero: ideal para NPU/DSP móvil")
    elif tamano_mb < 20:
        print("✓ Dentro del límite de 20MB para móvil")
    else:
        print("⚠ Mayor a 20MB. Considerar reducir capas del Transformer.")


if __name__ == "__main__":
    exportar_modelo_int8()
