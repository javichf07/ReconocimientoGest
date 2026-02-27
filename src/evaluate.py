import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from . import config
from . import data_loader

def evaluar_modelo():
    print("Cargando datos de prueba (usando set de validación por ahora)...")
    _, ds_validacion, nombres_clases = data_loader.cargar_datos()
    
    if ds_validacion is None:
        return

    print(f"Cargando modelo desde {config.RUTA_MEJOR_MODELO}...")
    try:
        modelo = tf.keras.models.load_model(config.RUTA_MEJOR_MODELO)
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    print("Evaluando modelo...")
    # Deshacer batch para obtener todas las etiquetas y predicciones
    y_verdadero = []
    y_predicho = []
    
    for imagenes, etiquetas in ds_validacion:
        preds = modelo.predict(imagenes, verbose=0)
        y_verdadero.extend(np.argmax(etiquetas.numpy(), axis=1))
        y_predicho.extend(np.argmax(preds, axis=1))
        
    y_verdadero = np.array(y_verdadero)
    y_predicho = np.array(y_predicho)
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_verdadero, y_predicho, target_names=nombres_clases))
    
    # Matriz de Confusión
    matriz_confusion = confusion_matrix(y_verdadero, y_predicho)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.xlabel('Predicho')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RUTA_MODELOS, 'confusion_matrix.png'))
    print(f"Matriz de Confusión guardada en {os.path.join(config.RUTA_MODELOS, 'confusion_matrix.png')}")

if __name__ == "__main__":
    evaluar_modelo()
