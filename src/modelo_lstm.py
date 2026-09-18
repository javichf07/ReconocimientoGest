"""
Modelo LSTM Bidireccional - Enfoque C

Arquitectura de red neuronal recurrente para clasificar
secuencias de landmarks de lenguaje de señas.

Input: (LONGITUD_SECUENCIA, NUM_LANDMARKS) = (30, 225)
Output: (num_clases,) con probabilidades softmax
"""

import tensorflow as tf
from . import config


def construir_modelo_lstm(num_clases, forma_entrada=(config.LONGITUD_SECUENCIA, config.NUM_LANDMARKS)):
    """
    Construye un modelo LSTM bidireccional para clasificación de secuencias
    de landmarks de lenguaje de señas.
    
    Arquitectura:
        Input (30, 225)
        → LSTM Bidireccional (128 unidades, return_sequences=True)
        → Dropout (0.3)
        → LSTM Bidireccional (64 unidades)
        → Dropout (0.3)
        → Dense (64, relu)
        → Dropout (0.3)
        → Dense (num_clases, softmax)
    
    Args:
        num_clases: número de señas/gestos a clasificar
        forma_entrada: tupla (longitud_secuencia, num_features)
    
    Returns:
        modelo Keras compilado
    """
    modelo = tf.keras.Sequential([
        # Capa de entrada
        tf.keras.layers.Input(shape=forma_entrada),
        
        # Primera capa LSTM Bidireccional
        # return_sequences=True para pasar la secuencia completa a la siguiente LSTM
        # Bidireccional: lee la secuencia hacia adelante Y hacia atrás
        # Esto ayuda a capturar patrones de movimiento en ambas direcciones
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(config.UNIDADES_LSTM_1, return_sequences=True)
        ),
        tf.keras.layers.Dropout(config.DROPOUT),
        
        # Segunda capa LSTM Bidireccional
        # return_sequences=False: solo retorna el último estado (resumen de toda la secuencia)
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(config.UNIDADES_LSTM_2)
        ),
        tf.keras.layers.Dropout(config.DROPOUT),
        
        # Capas densas para clasificación
        tf.keras.layers.Dense(config.UNIDADES_DENSA, activation='relu'),
        tf.keras.layers.Dropout(config.DROPOUT),
        
        # Capa de salida: una neurona por clase con softmax
        tf.keras.layers.Dense(num_clases, activation='softmax')
    ])
    
    return modelo
