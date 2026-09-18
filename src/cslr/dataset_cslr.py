"""
Dataset CSLR - Cargador de Datos
==================================
Procesa videos y genera secuencias de features normalizadas
para entrenamiento del modelo CSLR.

Pipeline:
    Videos → MediaPipe (Pose + Hands) → Normalización → Secuencias → .npy

Uso:
    python -m src.cslr.dataset_cslr
"""

import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from . import config_cslr as cfg
from .extraccion import ExtractorLandmarks
from .preprocesamiento import PreprocesadorCSLR


def extraer_features_video(ruta_video, extractor, preprocesador):
    """
    Extrae la secuencia completa de features CSLR de un video.
    
    Args:
        ruta_video: str, ruta al archivo de video
        extractor: ExtractorLandmarks inicializado
        preprocesador: PreprocesadorCSLR inicializado
        
    Returns:
        np.array (T, 275) secuencia de features, o None si falla
    """
    cap = cv2.VideoCapture(ruta_video)
    
    # Workaround para rutas con caracteres especiales en Windows
    if not cap.isOpened():
        import shutil
        ruta_temp = os.path.join(cfg._RAIZ_PROYECTO, "temp_video_cslr.mp4")
        try:
            shutil.copy2(ruta_video, ruta_temp)
            cap = cv2.VideoCapture(ruta_temp)
        except Exception as e:
            print(f"    Error abriendo video: {e}")
            return None
    
    if not cap.isOpened():
        return None
    
    frames_landmarks = []
    frames_con_mano = 0
    total_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        resultados = extractor.procesar_frame(frame)
        mano_izq, mano_der = extractor.separar_manos(resultados['manos'])
        
        frames_landmarks.append({
            'pose': resultados['pose'],
            'mano_izq': mano_izq,
            'mano_der': mano_der
        })
        
        if mano_izq is not None or mano_der is not None:
            frames_con_mano += 1
    
    cap.release()
    
    # Limpiar archivo temporal
    ruta_temp = os.path.join(cfg._RAIZ_PROYECTO, "temp_video_cslr.mp4")
    if os.path.exists(ruta_temp):
        try:
            os.remove(ruta_temp)
        except:
            pass
    
    if total_frames == 0:
        return None
    
    # Procesar secuencia completa
    secuencia = preprocesador.procesar_secuencia_desde_video(
        frames_landmarks, entrenamiento=True
    )
    
    deteccion = frames_con_mano / total_frames * 100 if total_frames > 0 else 0
    return secuencia, total_frames, deteccion


def crear_secuencias_ventana(features_video, longitud=cfg.LONGITUD_SECUENCIA_CSLR, 
                              paso=cfg.PASO_VENTANA_CSLR):
    """
    Divide las features de un video en secuencias de longitud fija
    con ventana deslizante y overlap.
    
    Args:
        features_video: np.array (T, 275)
        longitud: frames por secuencia
        paso: stride de la ventana
        
    Returns:
        lista de np.arrays, cada uno de shape (longitud, 275)
    """
    secuencias = []
    T = len(features_video)
    
    if T < longitud:
        # Padding con ceros si el video es muy corto
        padded = np.zeros((longitud, features_video.shape[1]), dtype=np.float32)
        padded[:T] = features_video
        secuencias.append(padded)
    else:
        for inicio in range(0, T - longitud + 1, paso):
            secuencia = features_video[inicio : inicio + longitud]
            secuencias.append(secuencia)
    
    return secuencias


def procesar_todos_los_videos():
    """
    Procesa todos los videos organizados por clase y genera
    archivos .npy de secuencias de features.
    
    Estructura esperada: videos/{CLASE}/*.mp4
    Salida: landmarks_cslr/{CLASE}/*.npy
    """
    if not os.path.exists(cfg.DIR_VIDEOS):
        print(f"Carpeta de videos no encontrada: {cfg.DIR_VIDEOS}")
        return
    
    clases = sorted([d for d in os.listdir(cfg.DIR_VIDEOS)
                     if os.path.isdir(os.path.join(cfg.DIR_VIDEOS, d))])
    
    if not clases:
        print("No se encontraron carpetas de clases en videos/")
        return
    
    print("=" * 60)
    print("EXTRACCIÓN DE FEATURES CSLR")
    print("=" * 60)
    print(f"Clases: {clases}")
    print(f"Secuencia: {cfg.LONGITUD_SECUENCIA_CSLR} frames, paso: {cfg.PASO_VENTANA_CSLR}")
    print(f"Dimensión feature: {cfg.DIM_FEATURE_FRAME}")
    print("=" * 60)
    
    if not os.path.exists(cfg.DIR_LANDMARKS_CSLR):
        os.makedirs(cfg.DIR_LANDMARKS_CSLR)
    
    extractor = ExtractorLandmarks()
    preprocesador = PreprocesadorCSLR()
    
    total_secuencias = 0
    
    for clase in clases:
        ruta_clase = os.path.join(cfg.DIR_VIDEOS, clase)
        ruta_destino = os.path.join(cfg.DIR_LANDMARKS_CSLR, clase)
        
        if not os.path.exists(ruta_destino):
            os.makedirs(ruta_destino)
        
        archivos = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            archivos.extend(glob.glob(os.path.join(ruta_clase, ext)))
        
        print(f"\nClase '{clase}': {len(archivos)} videos")
        sec_clase = 0
        
        for ruta_video in archivos:
            nombre = os.path.splitext(os.path.basename(ruta_video))[0]
            
            resultado = extraer_features_video(ruta_video, extractor, preprocesador)
            if resultado is None:
                print(f"  '{nombre}': error, saltando")
                continue
            
            features, total_f, deteccion = resultado
            
            if len(features) == 0:
                print(f"  '{nombre}': sin frames válidos")
                continue
            
            secuencias = crear_secuencias_ventana(features)
            
            for idx, seq in enumerate(secuencias):
                ruta_npy = os.path.join(ruta_destino, f"{nombre}_seq{idx}.npy")
                np.save(ruta_npy, seq.astype(np.float32))
            
            sec_clase += len(secuencias)
            print(f"  '{nombre}': {total_f} frames → {len(secuencias)} seqs "
                  f"(mano {deteccion:.0f}%)")
        
        total_secuencias += sec_clase
        print(f"  Total '{clase}': {sec_clase} secuencias")
    
    extractor.liberar()
    
    # Guardar labels
    with open(cfg.RUTA_LABELS_CSLR, 'w', encoding='utf-8') as f:
        for clase in clases:
            f.write(clase + '\n')
    
    print(f"\n{'='*60}")
    print(f"COMPLETADO: {total_secuencias} secuencias totales")
    print(f"Labels: {cfg.RUTA_LABELS_CSLR}")
    print(f"Datos: {cfg.DIR_LANDMARKS_CSLR}")


def cargar_datos(dir_datos=cfg.DIR_LANDMARKS_CSLR, split=0.2):
    """
    Carga secuencias .npy y divide en train/val.
    
    Returns:
        X_train, X_val: (N, T, 275)
        y_train, y_val: (N, num_clases) one-hot
        nombres_clases: lista de strings
    """
    if not os.path.exists(dir_datos):
        print(f"Datos no encontrados: {dir_datos}")
        print("Ejecuta: python -m src.cslr.dataset_cslr")
        return None, None, None, None, None
    
    nombres_clases = sorted([d for d in os.listdir(dir_datos)
                             if os.path.isdir(os.path.join(dir_datos, d))])
    
    if not nombres_clases:
        print("Sin clases en landmarks_cslr/")
        return None, None, None, None, None
    
    secuencias = []
    etiquetas = []
    
    for idx, clase in enumerate(nombres_clases):
        ruta = os.path.join(dir_datos, clase)
        archivos = [f for f in os.listdir(ruta) if f.endswith('.npy')]
        
        print(f"  Clase '{clase}': {len(archivos)} secuencias")
        
        for archivo in archivos:
            seq = np.load(os.path.join(ruta, archivo))
            if seq.shape == (cfg.LONGITUD_SECUENCIA_CSLR, cfg.DIM_FEATURE_FRAME):
                secuencias.append(seq)
                etiquetas.append(idx)
    
    if not secuencias:
        print("Sin secuencias válidas.")
        return None, None, None, None, None
    
    X = np.array(secuencias, dtype=np.float32)
    y = np.array(etiquetas)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=split, random_state=42, stratify=y
    )
    
    num_clases = len(nombres_clases)
    y_train_oh = np.eye(num_clases)[y_train].astype(np.float32)
    y_val_oh = np.eye(num_clases)[y_val].astype(np.float32)
    
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Clases: {num_clases}")
    
    return X_train, X_val, y_train_oh, y_val_oh, nombres_clases


if __name__ == "__main__":
    procesar_todos_los_videos()
