import tensorflow as tf
import numpy as np
from .. import config_cslr as cfg
from .atencion_euclidiana import AtencionEuclidiana

def codificacion_posicional(posicion, d_model):
    """Calcula la codificación posicional sinusoidal."""
    def get_angles(pos, i, d_model):
        tasas_angulo = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * tasas_angulo

    angulos = get_angles(np.arange(posicion)[:, np.newaxis],
                         np.arange(d_model)[np.newaxis, :],
                         d_model)

    angulos[:, 0::2] = np.sin(angulos[:, 0::2])
    angulos[:, 1::2] = np.cos(angulos[:, 1::2])

    pos_encoding = angulos[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def mascara_causal(seq_len):
    """Crea una máscara causal triangular superior para ocultar tokens futuros."""
    mascara = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mascara  # (seq_len, seq_len)

class CapaDecodificador(tf.keras.layers.Layer):
    """Capa individual del decodificador Transformer."""
    def __init__(self, dim_modelo, num_cabezas, dim_ffn, dropout=0.1, **kwargs):
        super(CapaDecodificador, self).__init__(**kwargs)
        self.atencion = AtencionEuclidiana(num_cabezas, dim_modelo)
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_ffn, activation='relu'),
            tf.keras.layers.Dense(dim_modelo)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, mask, training=False):
        # Auto-atención enmascarada
        atencion_salida = self.atencion(v=x, k=x, q=x, mask=mask)
        atencion_salida = self.dropout1(atencion_salida, training=training)
        salida1 = self.layernorm1(x + atencion_salida)
        
        # Red FFN
        ffn_salida = self.ffn(salida1)
        ffn_salida = self.dropout2(ffn_salida, training=training)
        salida2 = self.layernorm2(salida1 + ffn_salida)
        
        return salida2

class DecodificadorTransformer(tf.keras.layers.Layer):
    """
    Decodificador Transformer Solo (Decoder-Only) que utiliza atención euclidiana.
    """
    def __init__(self, num_clases, num_capas=cfg.NUM_CAPAS_TRANSFORMER,
                 dim_modelo=cfg.DIM_PROYECCION, num_cabezas=cfg.NUM_CABEZAS_ATENCION,
                 dim_ffn=cfg.DIM_FFN, dropout=cfg.DROPOUT_TRANSFORMER, max_pos=2048, **kwargs):
        super(DecodificadorTransformer, self).__init__(**kwargs)
        self.dim_modelo = dim_modelo
        self.num_capas = num_capas
        
        self.pos_encoding = codificacion_posicional(max_pos, dim_modelo)
        
        self.capas = [CapaDecodificador(dim_modelo, num_cabezas, dim_ffn, dropout) 
                      for _ in range(num_capas)]
        
        self.dropout = tf.keras.layers.Dropout(dropout)
        
        # Capa densa final para mapeo directo a tokens sin CTC
        self.capa_salida = tf.keras.layers.Dense(num_clases, activation='softmax')
        
    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        
        # Sumar codificación posicional a la secuencia de entrada
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        # Crear máscara causal
        mask = mascara_causal(seq_len)
        mask = mask[tf.newaxis, tf.newaxis, :, :]  # Expandir para (batch, num_cabezas, seq_len, seq_len)
        
        # Aplicar capas del decodificador
        for i in range(self.num_capas):
            x = self.capas[i](x, mask, training=training)
            
        # Clasificación
        salida = self.capa_salida(x)
        return salida
