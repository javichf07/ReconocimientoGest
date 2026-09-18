"""
Pipeline de Entrenamiento CSLR
================================
Entrena el modelo completo (1D-CNN Encoder + Transformer Decoder)
con los datos de secuencias de landmarks.

Uso:
    python -m src.cslr.entrenamiento
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from . import config_cslr as cfg
from . import dataset_cslr
from .modelo.modelo_cslr import construir_modelo_cslr, preparar_entradas


def graficar_historial(historial):
    """Guarda gráficos de entrenamiento."""
    acc = historial.history['accuracy']
    val_acc = historial.history['val_accuracy']
    loss = historial.history['loss']
    val_loss = historial.history['val_loss']
    epocas = range(len(acc))
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epocas, acc, 'b-', label='Train Accuracy')
    plt.plot(epocas, val_acc, 'r-', label='Val Accuracy')
    plt.title('Exactitud (CSLR)')
    plt.xlabel('Épocas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epocas, loss, 'b-', label='Train Loss')
    plt.plot(epocas, val_loss, 'r-', label='Val Loss')
    plt.title('Pérdida (CSLR)')
    plt.xlabel('Épocas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = os.path.join(cfg.DIR_MODELOS, 'cslr_training_history.png')
    plt.savefig(ruta, dpi=150)
    print(f"Historial guardado: {ruta}")


class GeneradorDatos(tf.keras.utils.Sequence):
    """
    Generador de datos que separa las features en los 3 flujos
    (pose, manos, cinético) en cada batch.
    """
    
    def __init__(self, X, y, batch_size=cfg.TAMANO_LOTE_CSLR, shuffle=True):
        self.X = X  # (N, T, 275)
        self.y = y  # (N, num_clases) one-hot
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx]
        y_batch = self.y[batch_idx]
        
        # Separar en los 3 flujos
        entradas = preparar_entradas(X_batch)
        
        return entradas, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def ejecutar_entrenamiento():
    """Pipeline principal de entrenamiento CSLR."""
    print("=" * 60)
    print("ENTRENAMIENTO CSLR")
    print("1D-CNN Encoder + Transformer Decoder")
    print("=" * 60)
    
    # 1. Cargar datos
    resultado = dataset_cslr.cargar_datos()
    if resultado[0] is None:
        return
    
    X_train, X_val, y_train, y_val, nombres_clases = resultado
    num_clases = len(nombres_clases)
    
    print(f"\nClases ({num_clases}): {nombres_clases}")
    print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
    
    # 2. Crear generadores
    gen_train = GeneradorDatos(X_train, y_train, shuffle=True)
    gen_val = GeneradorDatos(X_val, y_val, shuffle=False)
    
    # 3. Construir modelo
    print("\nConstruyendo modelo CSLR...")
    modelo = construir_modelo_cslr(
        num_clases=num_clases,
        longitud_secuencia=cfg.LONGITUD_SECUENCIA_CSLR
    )
    
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=cfg.TASA_APRENDIZAJE_CSLR
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.LABEL_SMOOTHING
        ),
        metrics=['accuracy']
    )
    
    modelo.summary()
    
    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            cfg.RUTA_MEJOR_MODELO_CSLR,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 5. Entrenar
    print("\n" + "=" * 60)
    print("INICIANDO ENTRENAMIENTO...")
    print("=" * 60)
    
    historial = modelo.fit(
        gen_train,
        validation_data=gen_val,
        epochs=cfg.EPOCAS_CSLR,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Guardar modelo completo + pesos por separado (para exportación robusta)
    modelo.save(cfg.RUTA_MODELO_CSLR)
    print(f"\nModelo guardado: {cfg.RUTA_MODELO_CSLR}")
    
    # Guardar pesos por separado para evitar problemas de deserialización
    # de capas custom al exportar a TFLite
    ruta_pesos = os.path.join(cfg.DIR_MODELOS, 'cslr_model_weights.h5')
    modelo.save_weights(ruta_pesos)
    print(f"Pesos guardados: {ruta_pesos}")
    
    graficar_historial(historial)
    
    mejor_acc = max(historial.history['val_accuracy'])
    print(f"\nMejor val_accuracy: {mejor_acc:.4f} ({mejor_acc*100:.1f}%)")
    
    if mejor_acc >= 0.85:
        print("✓ Objetivo de 85% ALCANZADO")
    else:
        print("⚠ Precisión < 85%. Agregar más videos o ajustar hiperparámetros.")


if __name__ == "__main__":
    ejecutar_entrenamiento()
