import tensorflow as tf
import os
from . import config

def cargar_datos(dir_datos=config.DIR_DATOS, tamano_lote=config.TAMANO_LOTE, tamano_img=config.TAMANO_IMG, division_validacion=0.2):
    """
    Carga datos del directorio, divide en entrenamiento/validación y aplica aumentación.
    """
    
    if not os.path.exists(dir_datos):
        print(f"Directorio del dataset no encontrado en: {dir_datos}")
        print("Por favor crea el directorio y añade tus carpetas de clases dentro.")
        return None, None, None

    # Cargar dataset (Entrenamiento)
    ds_entrenamiento = tf.keras.utils.image_dataset_from_directory(
        dir_datos,
        validation_split=division_validacion,
        subset="training",
        seed=123,
        image_size=tamano_img,
        batch_size=tamano_lote,
        label_mode='categorical',
        crop_to_aspect_ratio=True # Cortar centro en lugar de estirar
    )

    # Cargar dataset (Validación)
    ds_validacion = tf.keras.utils.image_dataset_from_directory(
        dir_datos,
        validation_split=division_validacion,
        subset="validation",
        seed=123,
        image_size=tamano_img,
        batch_size=tamano_lote,
        label_mode='categorical',
        crop_to_aspect_ratio=True
    )
    
    nombres_clases = ds_entrenamiento.class_names
    config.NUM_CLASES = len(nombres_clases)
    print(f"Encontradas {config.NUM_CLASES} clases: {nombres_clases}")

    # Aumentación de Datos para Entrenamiento
    aumentacion_datos = tf.keras.Sequential([
        #tf.keras.layers.RandomFlip("horizontal"), # PRECAUCIÓN: Evaluar si la distinción mano izquierda/derecha importa
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    # Preprocesamiento: Reescalado de valores de píxeles [-1, 1] para MobileNetV2
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    def preprocesar_entrenamiento(imagenes, etiquetas):
        imagenes = aumentacion_datos(imagenes)
        imagenes = preprocess_input(imagenes)
        return imagenes, etiquetas

    def preprocesar_validacion(imagenes, etiquetas):
        imagenes = preprocess_input(imagenes)
        return imagenes, etiquetas

    # Optimizar rendimiento del pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    ds_entrenamiento = ds_entrenamiento.map(preprocesar_entrenamiento, num_parallel_calls=AUTOTUNE)
    ds_validacion = ds_validacion.map(preprocesar_validacion, num_parallel_calls=AUTOTUNE)
    
    ds_entrenamiento = ds_entrenamiento.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    ds_validacion = ds_validacion.prefetch(buffer_size=AUTOTUNE)

    return ds_entrenamiento, ds_validacion, nombres_clases
