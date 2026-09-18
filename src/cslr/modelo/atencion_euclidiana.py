import tensorflow as tf
from .. import config_cslr as cfg

class AtencionEuclidiana(tf.keras.layers.Layer):
    """
    Capa de Atención Multi-Cabeza que utiliza la distancia euclidiana al cuadrado negativa
    en lugar del producto punto estándar para calcular los puntajes de atención.
    """
    def __init__(self, num_cabezas, dim_modelo, **kwargs):
        super(AtencionEuclidiana, self).__init__(**kwargs)
        self.num_cabezas = num_cabezas
        self.dim_modelo = dim_modelo
        
        assert dim_modelo % self.num_cabezas == 0, "dim_modelo debe ser divisible por num_cabezas"
        self.profundidad = dim_modelo // self.num_cabezas
        
        self.wq = tf.keras.layers.Dense(dim_modelo, use_bias=False)
        self.wk = tf.keras.layers.Dense(dim_modelo, use_bias=False)
        self.wv = tf.keras.layers.Dense(dim_modelo, use_bias=False)
        
        self.densa = tf.keras.layers.Dense(dim_modelo)
        
    def separar_cabezas(self, x, batch_size):
        """Separa la última dimensión en (num_cabezas, profundidad)."""
        x = tf.reshape(x, (batch_size, -1, self.num_cabezas, self.profundidad))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask=None):
        """
        Calcula la atención euclidiana.
        """
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len_q, dim_modelo)
        k = self.wk(k)  # (batch_size, seq_len_k, dim_modelo)
        v = self.wv(v)  # (batch_size, seq_len_v, dim_modelo)
        
        q = self.separar_cabezas(q, batch_size)  # (batch_size, num_cabezas, seq_len_q, profundidad)
        k = self.separar_cabezas(k, batch_size)  # (batch_size, num_cabezas, seq_len_k, profundidad)
        v = self.separar_cabezas(v, batch_size)  # (batch_size, num_cabezas, seq_len_v, profundidad)
        
        # Calcular distancia euclidiana al cuadrado: ||q - k||^2 = ||q||^2 + ||k||^2 - 2*q*k
        # q_sq: (batch_size, num_cabezas, seq_len_q, 1)
        q_sq = tf.reduce_sum(tf.square(q), axis=-1, keepdims=True)
        # k_sq: (batch_size, num_cabezas, 1, seq_len_k)
        k_sq = tf.expand_dims(tf.reduce_sum(tf.square(k), axis=-1), axis=-2)
        # qk: (batch_size, num_cabezas, seq_len_q, seq_len_k)
        qk = tf.matmul(q, k, transpose_b=True)
        
        # puntajes = -||q - k||^2
        puntajes_atencion = -(q_sq + k_sq - 2.0 * qk)
        
        # Aplicar la máscara si se proporciona (máscara causal)
        if mask is not None:
            # Se asume que la máscara tiene 1s en las posiciones a enmascarar
            puntajes_atencion += (mask * -1e9)
            
        pesos_atencion = tf.nn.softmax(puntajes_atencion, axis=-1)
        
        salida = tf.matmul(pesos_atencion, v)  # (batch_size, num_cabezas, seq_len_q, profundidad)
        salida = tf.transpose(salida, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_cabezas, profundidad)
        salida = tf.reshape(salida, (batch_size, -1, self.dim_modelo))  # (batch_size, seq_len_q, dim_modelo)
        
        salida = self.densa(salida)
        return salida
