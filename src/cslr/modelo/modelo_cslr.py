import tensorflow as tf
import numpy as np

from .. import config_cslr as cfg
from .codificador_cnn1d import CodificadorCNN1D
from .decodificador_transformer import DecodificadorTransformer

def construir_modelo_cslr(num_clases, longitud_secuencia=None):
    """
    Construye el modelo completo de Reconocimiento Continuo de Lenguaje de Señas (CSLR).
    
    Args:
        num_clases (int): Número total de clases/señas a predecir.
        longitud_secuencia (int, opcional): Longitud de la secuencia temporal. Por defecto es None (variable).
        
    Returns:
        tf.keras.Model: Modelo de Keras configurado para CSLR.
    """
    # Definir las entradas del modelo (shape no incluye el batch)
    input_pose = tf.keras.Input(shape=(longitud_secuencia, cfg.DIM_FLUJO_POSE), name='pose')
    input_manos = tf.keras.Input(shape=(longitud_secuencia, cfg.DIM_FLUJO_MANOS), name='manos')
    input_cinetico = tf.keras.Input(shape=(longitud_secuencia, cfg.DIM_FLUJO_CINETICO), name='cinetico')
    
    # 1. Pasar por el Codificador CNN 1D para obtener la representación codificada (batch, T, 128)
    codificador = CodificadorCNN1D()
    representacion_codificada = codificador({'pose': input_pose, 'manos': input_manos, 'cinetico': input_cinetico})
    
    # 2. Pasar por el Decodificador Transformer para procesar la secuencia temporal (batch, T, num_clases)
    decodificador = DecodificadorTransformer(num_clases=num_clases)
    salida_secuencia = decodificador(representacion_codificada)
    
    # 3. Aplicar agrupamiento promedio global (GlobalAveragePooling1D) en el tiempo (batch, num_clases)
    salida_final = tf.keras.layers.GlobalAveragePooling1D()(salida_secuencia)
    
    # 4. Construir y devolver el modelo
    modelo = tf.keras.Model(
        inputs=[input_pose, input_manos, input_cinetico],
        outputs=salida_final,
        name='modelo_cslr'
    )
    
    return modelo

def preparar_entradas(secuencia_features):
    """
    Toma un tensor de características y lo divide en los tres flujos de entrada: pose, manos y cinético.
    
    Args:
        secuencia_features (np.ndarray): Tensor de características en bruto con forma (batch, T, 275).
        
    Returns:
        dict: Diccionario con los 3 tensores listos para el modelo:
            - 'pose': (batch, T, 99)
            - 'manos': (batch, T, 176)
            - 'cinetico': (batch, T, 275)
    """
    # Extraer las partes estáticas (pose y manos)
    pose = secuencia_features[:, :, :cfg.DIM_FLUJO_POSE]
    manos = secuencia_features[:, :, cfg.DIM_FLUJO_POSE:cfg.DIM_FLUJO_POSE + cfg.DIM_FLUJO_MANOS]
    
    # Calcular el flujo cinético (diferencia temporal)
    # secuencia[:, 1:, :] - secuencia[:, :-1, :] con relleno de ceros al inicio
    cinetico = np.zeros_like(secuencia_features)
    if secuencia_features.shape[1] > 1:
        cinetico[:, 1:, :] = secuencia_features[:, 1:, :] - secuencia_features[:, :-1, :]
        
    return {
        'pose': pose,
        'manos': manos,
        'cinetico': cinetico
    }
