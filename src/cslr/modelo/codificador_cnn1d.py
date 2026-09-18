import tensorflow as tf
from tensorflow.keras import layers, Model
from .. import config_cslr as cfg

class CodificadorCNN1D(Model):
    """
    Codificador CNN 1D para procesar secuencias de características de lengua de señas.
    Utiliza tres flujos paralelos (pose, manos, cinético) para extraer características
    espaciales a lo largo del tiempo.
    """
    def __init__(self, **kwargs):
        super(CodificadorCNN1D, self).__init__(**kwargs)
        
        # Flujos individuales construidos dinámicamente
        self.flujo_pose = self.construir_flujo('pose')
        self.flujo_manos = self.construir_flujo('manos')
        self.flujo_cinetico = self.construir_flujo('cinetico')
        
        # Capa de proyección y normalización final
        self.proyeccion = layers.Dense(cfg.DIM_PROYECCION, name='proyeccion_densa')
        self.layer_norm = layers.LayerNormalization(name='layer_norm_proyeccion')
        
    def construir_flujo(self, prefijo):
        """
        Construye un flujo (stream) utilizando una secuencia de capas convolucionales y de normalización.
        
        Args:
            prefijo: Nombre base para las capas de este flujo.
            
        Returns:
            Modelo Sequential de Keras con las capas del flujo.
        """
        return tf.keras.Sequential([
            layers.Conv1D(cfg.FILTROS_CNN_1, kernel_size=cfg.KERNEL_CNN, activation='relu', padding='same', name=f'conv1_{prefijo}'),
            layers.BatchNormalization(name=f'bn1_{prefijo}'),
            layers.Conv1D(cfg.FILTROS_CNN_2, kernel_size=cfg.KERNEL_CNN, activation='relu', padding='same', name=f'conv2_{prefijo}'),
            layers.BatchNormalization(name=f'bn2_{prefijo}')
        ], name=f'flujo_{prefijo}')

    def call(self, inputs, training=False):
        """
        Procesa las entradas a través de los tres flujos paralelos.
        
        Args:
            inputs: Diccionario con tensores 'pose', 'manos', 'cinetico'.
            training: Booleano, indica si está en fase de entrenamiento (para BatchNorm).
            
        Returns:
            Tensor de salida con forma (batch, T, DIM_PROYECCION).
        """
        # Procesar cada flujo
        out_pose = self.flujo_pose(inputs['pose'], training=training)
        out_manos = self.flujo_manos(inputs['manos'], training=training)
        out_cinetico = self.flujo_cinetico(inputs['cinetico'], training=training)
        
        # Concatenación de características a lo largo del eje de características
        # (batch, T, 128) + (batch, T, 128) + (batch, T, 128) -> (batch, T, 384)
        concatenado = tf.concat([out_pose, out_manos, out_cinetico], axis=-1)
        
        # Proyección y normalización a la dimensión final
        proyectado = self.proyeccion(concatenado)
        salida = self.layer_norm(proyectado)
        
        return salida
        
    def get_config(self):
        """
        Obtiene la configuración para la serialización del modelo.
        """
        config = super(CodificadorCNN1D, self).get_config()
        return config


def construir_codificador(seq_length=None):
    """
    Construye y retorna el modelo CodificadorCNN1D utilizando la API Funcional de Keras.
    Esta representación es útil para visualizar la topología del modelo y facilita 
    significativamente la conversión a formato TFLite para inferencia.
    
    Args:
        seq_length: Longitud de la secuencia de tiempo (T). Si es None, acepta secuencias variables.
        
    Returns:
        Modelo Keras compilado con la API Funcional.
    """
    # Definir capas de entrada explícitas
    entrada_pose = layers.Input(shape=(seq_length, cfg.DIM_FLUJO_POSE), name='pose')
    entrada_manos = layers.Input(shape=(seq_length, cfg.DIM_FLUJO_MANOS), name='manos')
    entrada_cinetico = layers.Input(shape=(seq_length, cfg.DIM_FLUJO_CINETICO), name='cinetico')
    
    # Función auxiliar interna para construir los flujos funcionales
    def aplicar_flujo(x, prefijo):
        x = layers.Conv1D(cfg.FILTROS_CNN_1, kernel_size=cfg.KERNEL_CNN, activation='relu', padding='same', name=f'conv1_{prefijo}')(x)
        x = layers.BatchNormalization(name=f'bn1_{prefijo}')(x)
        x = layers.Conv1D(cfg.FILTROS_CNN_2, kernel_size=cfg.KERNEL_CNN, activation='relu', padding='same', name=f'conv2_{prefijo}')(x)
        x = layers.BatchNormalization(name=f'bn2_{prefijo}')(x)
        return x
        
    # Procesar flujos paralelos
    flujo_pose = aplicar_flujo(entrada_pose, 'pose')
    flujo_manos = aplicar_flujo(entrada_manos, 'manos')
    flujo_cinetico = aplicar_flujo(entrada_cinetico, 'cinetico')
    
    # Concatenar los tres flujos
    concatenado = layers.Concatenate(axis=-1, name='concatenacion_flujos')([flujo_pose, flujo_manos, flujo_cinetico])
    
    # Proyección densa y normalización final
    proyectado = layers.Dense(cfg.DIM_PROYECCION, name='proyeccion_densa')(concatenado)
    salida = layers.LayerNormalization(name='layer_norm_proyeccion')(proyectado)
    
    # Ensamblar modelo completo
    modelo = Model(
        inputs={'pose': entrada_pose, 'manos': entrada_manos, 'cinetico': entrada_cinetico},
        outputs=salida,
        name='codificador_cnn1d_funcional'
    )
    
    return modelo
