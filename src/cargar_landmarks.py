"""
Cargador de Datos de Landmarks - Enfoque C

Carga los archivos .npy de secuencias de landmarks,
asigna etiquetas y divide en entrenamiento/validación.

Incluye aumentación de datos para mejorar la robustez del modelo.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from . import config


def aumentar_secuencia(secuencia):
    """
    Aplica aumentación de datos a una secuencia de landmarks.
    Genera variantes para aumentar el dataset.
    
    Args:
        secuencia: array de shape (LONGITUD_SECUENCIA, NUM_LANDMARKS)
    
    Returns:
        lista de secuencias aumentadas (incluyendo la original)
    """
    aumentadas = [secuencia]  # Siempre incluir la original
    
    # 1. Agregar ruido gaussiano pequeño (simula imprecisión del detector)
    ruido = secuencia + np.random.normal(0, 0.01, secuencia.shape)
    aumentadas.append(ruido)
    
    # 2. Escalar ligeramente (simula estar más cerca/lejos de la cámara)
    factor_escala = np.random.uniform(0.9, 1.1)
    escalada = secuencia * factor_escala
    aumentadas.append(escalada)
    
    # 3. Variación temporal: velocidad ligeramente diferente
    # Interpolar para hacer la secuencia un poco más rápida o lenta
    factor_tiempo = np.random.uniform(0.85, 1.15)
    num_frames_original = len(secuencia)
    num_frames_nuevo = int(num_frames_original * factor_tiempo)
    
    if num_frames_nuevo > 1:
        indices_originales = np.linspace(0, num_frames_original - 1, num_frames_nuevo)
        indices_objetivo = np.linspace(0, num_frames_original - 1, num_frames_original)
        
        temporal = np.zeros_like(secuencia)
        for feat in range(secuencia.shape[1]):
            temporal[:, feat] = np.interp(indices_objetivo, indices_originales,
                                          np.interp(indices_originales, 
                                                    range(num_frames_original), 
                                                    secuencia[:, feat]))
        aumentadas.append(temporal)
    
    return aumentadas


def cargar_datos(dir_landmarks=config.DIR_LANDMARKS, division_validacion=0.2, aumentar=True):
    """
    Carga todas las secuencias .npy del directorio de landmarks,
    asigna etiquetas y divide en entrenamiento/validación.
    
    Args:
        dir_landmarks: ruta al directorio con subcarpetas por clase
        division_validacion: porcentaje para validación (default 20%)
        aumentar: si True, aplica aumentación de datos al set de entrenamiento
    
    Returns:
        X_train, X_val: arrays de secuencias (n, LONGITUD_SECUENCIA, NUM_LANDMARKS)
        y_train, y_val: arrays de etiquetas one-hot (n, num_clases)
        nombres_clases: lista de nombres de clases
    """
    if not os.path.exists(dir_landmarks):
        print(f"Directorio de landmarks no encontrado: {dir_landmarks}")
        print("Ejecuta primero: python -m src.extraer_landmarks")
        return None, None, None, None, None
    
    # Obtener clases (subcarpetas ordenadas)
    nombres_clases = sorted([d for d in os.listdir(dir_landmarks) 
                             if os.path.isdir(os.path.join(dir_landmarks, d))])
    
    if not nombres_clases:
        print("No se encontraron carpetas de clases en landmarks_data/")
        return None, None, None, None, None
    
    print(f"Clases encontradas: {nombres_clases}")
    
    secuencias = []
    etiquetas = []
    
    for idx_clase, clase in enumerate(nombres_clases):
        ruta_clase = os.path.join(dir_landmarks, clase)
        archivos = [f for f in os.listdir(ruta_clase) if f.endswith('.npy')]
        
        print(f"  Clase '{clase}' (id={idx_clase}): {len(archivos)} secuencias")
        
        for archivo in archivos:
            ruta_archivo = os.path.join(ruta_clase, archivo)
            secuencia = np.load(ruta_archivo)
            
            # Verificar forma correcta
            if secuencia.shape == (config.LONGITUD_SECUENCIA, config.NUM_LANDMARKS):
                secuencias.append(secuencia)
                etiquetas.append(idx_clase)
            else:
                print(f"    Advertencia: {archivo} tiene forma {secuencia.shape}, esperada "
                      f"({config.LONGITUD_SECUENCIA}, {config.NUM_LANDMARKS}). Saltando.")
    
    if len(secuencias) == 0:
        print("No se cargaron secuencias válidas.")
        return None, None, None, None, None
    
    X = np.array(secuencias)
    y = np.array(etiquetas)
    
    print(f"\nTotal secuencias cargadas: {len(X)}")
    print(f"Forma de datos: {X.shape}")
    
    # Dividir en entrenamiento y validación con estratificación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=division_validacion, 
        random_state=42, 
        stratify=y  # Mantener proporción de clases
    )
    
    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Validación: {len(X_val)} muestras")
    
    # Aumentación de datos (solo para entrenamiento)
    if aumentar:
        print("\nAplicando aumentación de datos...")
        X_train_aumentado = []
        y_train_aumentado = []
        
        for i in range(len(X_train)):
            variantes = aumentar_secuencia(X_train[i])
            for variante in variantes:
                X_train_aumentado.append(variante)
                y_train_aumentado.append(y_train[i])
        
        X_train = np.array(X_train_aumentado)
        y_train = np.array(y_train_aumentado)
        print(f"Entrenamiento después de aumentación: {len(X_train)} muestras")
    
    # Convertir etiquetas a one-hot encoding
    num_clases = len(nombres_clases)
    y_train_onehot = np.eye(num_clases)[y_train]
    y_val_onehot = np.eye(num_clases)[y_val]
    
    return X_train, X_val, y_train_onehot, y_val_onehot, nombres_clases
