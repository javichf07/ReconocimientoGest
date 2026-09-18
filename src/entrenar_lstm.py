"""
Entrenamiento del Modelo LSTM - Enfoque C

Carga datos de landmarks, construye el modelo LSTM, lo entrena,
guarda el mejor modelo, genera gráficos de entrenamiento y exporta a TFLite.

Uso:
    python -m src.entrenar_lstm
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from . import config
from . import cargar_landmarks
from . import modelo_lstm


def graficar_historial(historial):
    """Guarda gráficos de exactitud y pérdida del entrenamiento."""
    exactitud = historial.history['accuracy']
    val_exactitud = historial.history['val_accuracy']
    perdida = historial.history['loss']
    val_perdida = historial.history['val_loss']
    
    epocas_rango = range(len(exactitud))
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epocas_rango, exactitud, 'b-', label='Exactitud Entrenamiento')
    plt.plot(epocas_rango, val_exactitud, 'r-', label='Exactitud Validación')
    plt.title('Exactitud de Entrenamiento y Validación (LSTM)')
    plt.xlabel('Épocas')
    plt.ylabel('Exactitud')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epocas_rango, perdida, 'b-', label='Pérdida Entrenamiento')
    plt.plot(epocas_rango, val_perdida, 'r-', label='Pérdida Validación')
    plt.title('Pérdida de Entrenamiento y Validación (LSTM)')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta_grafico = os.path.join(config.RUTA_MODELOS, 'lstm_training_history.png')
    plt.savefig(ruta_grafico, dpi=150)
    print(f"Historial de entrenamiento guardado en {ruta_grafico}")


def exportar_a_tflite(modelo):
    """Convierte el modelo Keras a TFLite con optimización."""
    print("\nExportando a TFLite...")
    convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Necesario para modelos LSTM/Bidireccional en TFLite
    convertidor.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Operaciones nativas TFLite
        tf.lite.OpsSet.SELECT_TF_OPS      # Operaciones de TensorFlow completo (para LSTM)
    ]
    convertidor._experimental_lower_tensor_list_ops = False
    
    modelo_tflite = convertidor.convert()
    
    with open(config.RUTA_MODELO_LSTM_TFLITE, 'wb') as f:
        f.write(modelo_tflite)
    
    tamano_mb = os.path.getsize(config.RUTA_MODELO_LSTM_TFLITE) / (1024 * 1024)
    print(f"Modelo TFLite guardado: {config.RUTA_MODELO_LSTM_TFLITE}")
    print(f"Tamaño: {tamano_mb:.2f} MB")
    
    if tamano_mb < 5:
        print("✓ Modelo ultra-ligero, perfecto para móviles")
    elif tamano_mb < 20:
        print("✓ Modelo dentro del límite de 20MB")
    else:
        print("⚠ Modelo supera 20MB, considerar reducir unidades LSTM")


def ejecutar_entrenamiento():
    """Función principal de entrenamiento."""
    print("=" * 60)
    print("ENTRENAMIENTO LSTM - ENFOQUE C (LANDMARKS)")
    print("=" * 60)
    
    # 1. Cargar datos
    resultado = cargar_landmarks.cargar_datos()
    
    if resultado[0] is None:
        return
    
    X_train, X_val, y_train, y_val, nombres_clases = resultado
    
    num_clases = len(nombres_clases)
    print(f"\nEntrenando con {num_clases} clases: {nombres_clases}")
    print(f"Forma X_train: {X_train.shape}")
    print(f"Forma X_val: {X_val.shape}")
    
    # 2. Construir modelo
    modelo = modelo_lstm.construir_modelo_lstm(num_clases=num_clases)
    
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.TASA_APRENDIZAJE_LSTM),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nArquitectura del modelo:")
    modelo.summary()
    
    # 3. Configurar callbacks
    retrollamadas = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.RUTA_MEJOR_MODELO_LSTM,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Más paciencia para LSTM
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    # 4. Entrenar
    print("\n" + "=" * 60)
    print("INICIANDO ENTRENAMIENTO...")
    print("=" * 60)
    
    historial = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCAS_LSTM,
        batch_size=config.TAMANO_LOTE_LSTM,
        callbacks=retrollamadas,
        verbose=1
    )
    
    # 5. Guardar modelo final
    modelo.save(config.RUTA_MODELO_FINAL_LSTM)
    print(f"\nModelo final guardado en {config.RUTA_MODELO_FINAL_LSTM}")
    
    # 6. Guardar gráfico de historial
    graficar_historial(historial)
    
    # 7. Guardar nombres de clases
    ruta_labels = os.path.join(config.RUTA_MODELOS, 'labels_lstm.txt')
    with open(ruta_labels, 'w', encoding='utf-8') as f:
        for nombre in nombres_clases:
            f.write(nombre + '\n')
    print(f"Etiquetas guardadas en {ruta_labels}")
    
    # 8. Exportar a TFLite
    exportar_a_tflite(modelo)
    
    # 9. Resumen final
    mejor_val_acc = max(historial.history['val_accuracy'])
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Mejor exactitud de validación: {mejor_val_acc:.4f} ({mejor_val_acc*100:.1f}%)")
    print("=" * 60)
    
    if mejor_val_acc >= 0.85:
        print("✓ Objetivo de 85% de precisión ALCANZADO")
    else:
        print("⚠ Precisión por debajo del 85%. Sugerencias:")
        print("  - Agregar más videos por clase")
        print("  - Aumentar EPOCAS_LSTM en config.py")
        print("  - Verificar calidad de los videos (buena iluminación, manos visibles)")


if __name__ == "__main__":
    ejecutar_entrenamiento()
