import tensorflow as tf
from . import config

def construir_modelo(num_clases=config.NUM_CLASES, forma_entrada=(config.ALTO_IMG, config.ANCHO_IMG, 3)):
    """
    Construye un modelo basado en MobileNetV2 para Reconocimiento de Lenguaje de Señas.
    """
    modelo_base = tf.keras.applications.MobileNetV2(
        input_shape=forma_entrada,
        include_top=False,
        weights='imagenet',
        alpha=config.MOBILENET_ALFA
    )
    
    # Congelar el modelo base primero
    modelo_base.trainable = False
    
    entradas = tf.keras.Input(shape=forma_entrada)
    x = modelo_base(entradas, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x) # Reducir sobreajuste (overfitting)
    salidas = tf.keras.layers.Dense(num_clases, activation='softmax')(x)
    
    modelo = tf.keras.Model(entradas, salidas)
    
    return modelo
