"""
Extractor de Landmarks - Enfoque C

Procesa los videos de cada clase y extrae landmarks (coordenadas de articulaciones)
usando MediaPipe Holistic. Guarda secuencias de 30 frames como archivos .npy.

Uso:
    python -m src.extraer_landmarks
"""

import cv2
import os
import glob
import numpy as np
import mediapipe as mp
from . import config


def inicializar_holistic():
    """Inicializa el modelo MediaPipe Holistic para detección de pose y manos."""
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # 0=lite, 1=full, 2=heavy
    )
    return holistic


def extraer_landmarks_frame(resultados):
    """
    Extrae y concatena los landmarks de pose, mano izquierda y mano derecha
    de un frame procesado por MediaPipe Holistic.
    
    Retorna un array de 225 valores (75 puntos × 3 coordenadas).
    Si algún grupo no se detecta, se rellena con ceros.
    """
    # Pose: 33 landmarks × 3 = 99 valores
    if resultados.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in resultados.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(config.NUM_PUNTOS_POSE * config.NUM_COORDENADAS)
    
    # Mano Izquierda: 21 landmarks × 3 = 63 valores
    if resultados.left_hand_landmarks:
        mano_izq = np.array([[lm.x, lm.y, lm.z] for lm in resultados.left_hand_landmarks.landmark]).flatten()
    else:
        mano_izq = np.zeros(config.NUM_PUNTOS_MANO * config.NUM_COORDENADAS)
    
    # Mano Derecha: 21 landmarks × 3 = 63 valores
    if resultados.right_hand_landmarks:
        mano_der = np.array([[lm.x, lm.y, lm.z] for lm in resultados.right_hand_landmarks.landmark]).flatten()
    else:
        mano_der = np.zeros(config.NUM_PUNTOS_MANO * config.NUM_COORDENADAS)
    
    # Concatenar todo: [pose(99) + mano_izq(63) + mano_der(63)] = 225
    return np.concatenate([pose, mano_izq, mano_der])


def normalizar_secuencia(secuencia):
    """
    Normaliza las coordenadas de una secuencia para que sean independientes
    de la posición y escala de la persona en la cámara.
    
    Estrategia:
    - Punto de referencia: centro entre los hombros (pose landmarks 11 y 12)
    - Factor de escala: distancia entre hombros
    - Todas las coordenadas se vuelven relativas al centro del cuerpo
    
    Args:
        secuencia: array de shape (num_frames, 225)
    
    Returns:
        secuencia normalizada de shape (num_frames, 225)
    """
    secuencia_norm = secuencia.copy()
    
    for i in range(len(secuencia_norm)):
        frame = secuencia_norm[i]
        
        # Extraer coordenadas de los hombros de la pose
        # Pose landmarks están en los primeros 99 valores (33 puntos × 3)
        # Hombro izquierdo = landmark 11 → índices [33, 34, 35] (11*3, 11*3+1, 11*3+2)
        # Hombro derecho = landmark 12 → índices [36, 37, 38]
        hombro_izq = frame[11*3 : 11*3+3]  # [x, y, z]
        hombro_der = frame[12*3 : 12*3+3]  # [x, y, z]
        
        # Verificar que los hombros están detectados (no son ceros)
        if np.all(hombro_izq == 0) and np.all(hombro_der == 0):
            # Si no hay pose detectada, no normalizar este frame
            continue
        
        # Centro de referencia: punto medio entre hombros
        centro = (hombro_izq + hombro_der) / 2.0
        
        # Factor de escala: distancia entre hombros
        distancia_hombros = np.linalg.norm(hombro_izq - hombro_der)
        if distancia_hombros < 0.001:  # Evitar división por cero
            distancia_hombros = 1.0
        
        # Normalizar cada grupo de 3 coordenadas (x, y, z)
        for j in range(0, len(frame), 3):
            punto = frame[j:j+3]
            if not np.all(punto == 0):  # Solo normalizar puntos detectados
                secuencia_norm[i, j:j+3] = (punto - centro) / distancia_hombros
    
    return secuencia_norm


def crear_secuencias(landmarks_video, longitud=config.LONGITUD_SECUENCIA, paso=config.PASO_VENTANA):
    """
    Divide los landmarks de un video completo en secuencias de longitud fija
    usando una ventana deslizante con overlap.
    
    Args:
        landmarks_video: array de shape (total_frames, 225)
        longitud: frames por secuencia (default 30)
        paso: avance de la ventana (default 15 = 50% overlap)
    
    Returns:
        lista de secuencias, cada una de shape (longitud, 225)
    """
    secuencias = []
    total_frames = len(landmarks_video)
    
    if total_frames < longitud:
        # Si el video es muy corto, rellenar con el último frame
        padding = np.zeros((longitud - total_frames, landmarks_video.shape[1]))
        landmarks_padded = np.vstack([landmarks_video, padding])
        secuencias.append(landmarks_padded)
    else:
        for inicio in range(0, total_frames - longitud + 1, paso):
            secuencia = landmarks_video[inicio : inicio + longitud]
            secuencias.append(secuencia)
    
    return secuencias


def procesar_videos():
    """
    Función principal: recorre todos los videos por clase,
    extrae landmarks, normaliza, crea secuencias y guarda como .npy
    """
    ruta_base_videos = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'videos')
    
    if not os.path.exists(ruta_base_videos):
        print(f"No se encontró la carpeta de videos en: {ruta_base_videos}")
        return
    
    # Listar clases (subcarpetas)
    clases = sorted([d for d in os.listdir(ruta_base_videos) 
                     if os.path.isdir(os.path.join(ruta_base_videos, d))])
    
    if not clases:
        print("No se encontraron carpetas de clases en 'videos/'")
        return
    
    print(f"Clases encontradas: {clases}")
    print(f"Configuración: secuencia={config.LONGITUD_SECUENCIA} frames, paso={config.PASO_VENTANA}")
    print("=" * 60)
    
    # Crear directorio de salida
    if not os.path.exists(config.DIR_LANDMARKS):
        os.makedirs(config.DIR_LANDMARKS)
    
    # Inicializar MediaPipe Holistic
    holistic = inicializar_holistic()
    
    total_secuencias = 0
    
    for clase in clases:
        ruta_clase_videos = os.path.join(ruta_base_videos, clase)
        ruta_clase_destino = os.path.join(config.DIR_LANDMARKS, clase)
        
        if not os.path.exists(ruta_clase_destino):
            os.makedirs(ruta_clase_destino)
        
        # Buscar archivos de video
        archivos_video = []
        extensiones = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        for ext in extensiones:
            archivos_video.extend(glob.glob(os.path.join(ruta_clase_videos, ext)))
        
        print(f"\nClase '{clase}': {len(archivos_video)} videos encontrados")
        
        secuencias_clase = 0
        
        for idx_video, ruta_video in enumerate(archivos_video):
            nombre_video = os.path.splitext(os.path.basename(ruta_video))[0]
            
            # Abrir video (con workaround para caracteres especiales)
            cap = cv2.VideoCapture(ruta_video)
            
            if not cap.isOpened():
                # Workaround para rutas con caracteres especiales en Windows
                import shutil
                ruta_temp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_video.mp4")
                try:
                    shutil.copy2(ruta_video, ruta_temp)
                    cap = cv2.VideoCapture(ruta_temp)
                except Exception as e:
                    print(f"  Error abriendo '{nombre_video}': {e}")
                    continue
            
            # Extraer landmarks de cada frame
            landmarks_video = []
            frames_procesados = 0
            frames_con_mano = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convertir BGR a RGB para MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False  # Optimización de rendimiento
                
                resultados = holistic.process(frame_rgb)
                
                # Extraer landmarks del frame
                landmarks_frame = extraer_landmarks_frame(resultados)
                landmarks_video.append(landmarks_frame)
                
                frames_procesados += 1
                
                # Contar frames donde se detectó al menos una mano
                if resultados.left_hand_landmarks or resultados.right_hand_landmarks:
                    frames_con_mano += 1
            
            cap.release()
            
            # Limpiar archivo temporal si existe
            ruta_temp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_video.mp4")
            if os.path.exists(ruta_temp):
                try:
                    os.remove(ruta_temp)
                except:
                    pass
            
            if len(landmarks_video) == 0:
                print(f"  Video '{nombre_video}': sin frames válidos, saltando.")
                continue
            
            landmarks_video = np.array(landmarks_video)  # shape: (total_frames, 225)
            
            # Normalizar coordenadas
            landmarks_video = normalizar_secuencia(landmarks_video)
            
            # Crear secuencias con ventana deslizante
            secuencias = crear_secuencias(landmarks_video)
            
            # Guardar cada secuencia como archivo .npy
            for idx_seq, secuencia in enumerate(secuencias):
                nombre_archivo = f"{nombre_video}_seq{idx_seq}.npy"
                ruta_archivo = os.path.join(ruta_clase_destino, nombre_archivo)
                np.save(ruta_archivo, secuencia)
            
            secuencias_clase += len(secuencias)
            deteccion_pct = (frames_con_mano / frames_procesados * 100) if frames_procesados > 0 else 0
            print(f"  Video '{nombre_video}': {frames_procesados} frames → {len(secuencias)} secuencias "
                  f"(mano detectada en {deteccion_pct:.0f}% frames)")
        
        total_secuencias += secuencias_clase
        print(f"  Total clase '{clase}': {secuencias_clase} secuencias guardadas")
    
    holistic.close()
    
    print("\n" + "=" * 60)
    print(f"EXTRACCIÓN COMPLETADA")
    print(f"Total secuencias generadas: {total_secuencias}")
    print(f"Datos guardados en: {config.DIR_LANDMARKS}")
    print(f"Cada secuencia tiene forma: ({config.LONGITUD_SECUENCIA}, {config.NUM_LANDMARKS})")


if __name__ == "__main__":
    procesar_videos()
