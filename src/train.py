import tensorflow as tf
import os
import matplotlib.pyplot as plt
from . import config
from . import data_loader
from . import model as constructor_modelo

def graficar_historial(historial):
    exactitud = historial.history['accuracy']
    val_exactitud = historial.history['val_accuracy']
    perdida = historial.history['loss']
    val_perdida = historial.history['val_loss']
    
    epocas_rango = range(len(exactitud))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epocas_rango, exactitud, 'b', label='Exactitud Entrenamiento')
    plt.plot(epocas_rango, val_exactitud, 'r', label='Exactitud Validación')
    plt.title('Exactitud de Entrenamiento y Validación')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epocas_rango, perdida, 'b', label='Pérdida Entrenamiento')
    plt.plot(epocas_rango, val_perdida, 'r', label='Pérdida Validación')
    plt.title('Pérdida de Entrenamiento y Validación')
    plt.legend()
    
    plt.savefig(os.path.join(config.RUTA_MODELOS, 'training_history.png'))
    print(f"Historial de entrenamiento guardado en {os.path.join(config.RUTA_MODELOS, 'training_history.png')}")

def ejecutar_entrenamiento():
    ds_entrenamiento, ds_validacion, nombres_clases = data_loader.cargar_datos()
    
    if ds_entrenamiento is None:
        return

    num_clases = len(nombres_clases)
    print(f"Entrenando en {num_clases} clases.")

    modelo = constructor_modelo.construir_modelo(num_clases=num_clases)
    
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.TASA_APRENDIZAJE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    modelo.summary()
    
    retrollamadas = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.RUTA_MEJOR_MODELO,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    print("Iniciando entrenamiento...")
    historial = modelo.fit(
        ds_entrenamiento,
        validation_data=ds_validacion,
        epochs=config.EPOCAS,
        callbacks=retrollamadas
    )
    
    # Guardar modelo final
    modelo.save(config.RUTA_MODELO_FINAL)
    print(f"Modelo guardado en {config.RUTA_MODELO_FINAL}")
    
    graficar_historial(historial)
    
    # Guardar nombres de clases en un archivo para uso posterior
    with open(os.path.join(config.RUTA_MODELOS, 'labels.txt'), 'w') as f:
        for nombre in nombres_clases:
            f.write(nombre + '\n')
            
if __name__ == "__main__":
    ejecutar_entrenamiento()
