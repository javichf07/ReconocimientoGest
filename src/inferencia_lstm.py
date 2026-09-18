"""
Inferencia en Tiempo Real con LSTM - Enfoque C

Usa MediaPipe Holistic para extraer landmarks del cuerpo y manos en tiempo real,
mantiene un buffer de 30 frames, y clasifica la secuencia con el modelo LSTM.

Dibuja el esqueleto de pose y manos, predicción y oración en pantalla.

Uso:
    python -m src.inferencia_lstm
"""

import cv2
import tensorflow as tf
import numpy as np
import time
import os
import mediapipe as mp
from collections import deque
from . import config
from .realtime_processor import ProcesadorTiempoReal


def extraer_landmarks_frame(resultados):
    """
    Extrae landmarks de pose, mano izquierda y mano derecha de un frame.
    Retorna array de 225 valores. Rellena con ceros si no se detecta.
    """
    # Pose: 33 × 3 = 99
    if resultados.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] 
                         for lm in resultados.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(config.NUM_PUNTOS_POSE * config.NUM_COORDENADAS)
    
    # Mano Izquierda: 21 × 3 = 63
    if resultados.left_hand_landmarks:
        mano_izq = np.array([[lm.x, lm.y, lm.z] 
                             for lm in resultados.left_hand_landmarks.landmark]).flatten()
    else:
        mano_izq = np.zeros(config.NUM_PUNTOS_MANO * config.NUM_COORDENADAS)
    
    # Mano Derecha: 21 × 3 = 63
    if resultados.right_hand_landmarks:
        mano_der = np.array([[lm.x, lm.y, lm.z] 
                             for lm in resultados.right_hand_landmarks.landmark]).flatten()
    else:
        mano_der = np.zeros(config.NUM_PUNTOS_MANO * config.NUM_COORDENADAS)
    
    return np.concatenate([pose, mano_izq, mano_der])


def normalizar_frame(frame_landmarks, secuencia_buffer):
    """
    Normaliza un frame individual usando los hombros como referencia.
    Misma lógica que en extraer_landmarks.py para consistencia.
    """
    frame = frame_landmarks.copy()
    
    # Hombro izquierdo (landmark 11) y derecho (landmark 12) de pose
    hombro_izq = frame[11*3 : 11*3+3]
    hombro_der = frame[12*3 : 12*3+3]
    
    if np.all(hombro_izq == 0) and np.all(hombro_der == 0):
        return frame  # Sin pose detectada, no normalizar
    
    centro = (hombro_izq + hombro_der) / 2.0
    distancia_hombros = np.linalg.norm(hombro_izq - hombro_der)
    
    if distancia_hombros < 0.001:
        distancia_hombros = 1.0
    
    for j in range(0, len(frame), 3):
        punto = frame[j:j+3]
        if not np.all(punto == 0):
            frame[j:j+3] = (punto - centro) / distancia_hombros
    
    return frame


def dibujar_landmarks(frame, resultados, mp_holistic, mp_dibujo):
    """Dibuja los landmarks de pose y manos sobre el frame."""
    # Dibujar Pose (esqueleto del cuerpo)
    if resultados.pose_landmarks:
        mp_dibujo.draw_landmarks(
            frame,
            resultados.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_dibujo.DrawingSpec(color=(80, 110, 200), thickness=2, circle_radius=2),
            mp_dibujo.DrawingSpec(color=(80, 180, 200), thickness=2, circle_radius=1)
        )
    
    # Dibujar Mano Izquierda (verde)
    if resultados.left_hand_landmarks:
        mp_dibujo.draw_landmarks(
            frame,
            resultados.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_dibujo.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_dibujo.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=2)
        )
    
    # Dibujar Mano Derecha (rojo)
    if resultados.right_hand_landmarks:
        mp_dibujo.draw_landmarks(
            frame,
            resultados.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_dibujo.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
            mp_dibujo.DrawingSpec(color=(0, 0, 200), thickness=2, circle_radius=2)
        )


def ejecutar_inferencia():
    """Función principal de inferencia en tiempo real."""
    
    # 1. Cargar modelo TFLite
    if not os.path.exists(config.RUTA_MODELO_LSTM_TFLITE):
        print(f"Modelo LSTM no encontrado en {config.RUTA_MODELO_LSTM_TFLITE}")
        print("Ejecuta primero: python -m src.entrenar_lstm")
        return
    
    print("Cargando modelo LSTM TFLite...")
    interprete = tf.lite.Interpreter(model_path=config.RUTA_MODELO_LSTM_TFLITE)
    interprete.allocate_tensors()
    
    detalles_entrada = interprete.get_input_details()
    detalles_salida = interprete.get_output_details()
    
    print(f"Forma de entrada del modelo: {detalles_entrada[0]['shape']}")
    
    # 2. Cargar etiquetas
    ruta_etiquetas = os.path.join(config.RUTA_MODELOS, 'labels_lstm.txt')
    if not os.path.exists(ruta_etiquetas):
        print("Archivo de etiquetas no encontrado. Ejecuta entrenar_lstm.py primero.")
        return
    
    with open(ruta_etiquetas, 'r', encoding='utf-8') as f:
        nombres_clases = [line.strip() for line in f.readlines()]
    
    print(f"Clases: {nombres_clases}")
    
    # 3. Inicializar MediaPipe Holistic
    print("Inicializando MediaPipe Holistic...")
    mp_holistic = mp.solutions.holistic
    mp_dibujo = mp.solutions.drawing_utils
    
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    # 4. Inicializar procesador de oraciones
    procesador = ProcesadorTiempoReal()
    
    # 5. Buffer circular para acumular frames de landmarks
    buffer_secuencia = deque(maxlen=config.LONGITUD_SECUENCIA)
    
    # 6. Abrir webcam
    captura = cv2.VideoCapture(0)
    
    if not captura.isOpened():
        print("No se puede abrir la cámara")
        return
    
    print("\n" + "=" * 50)
    print("INFERENCIA LSTM EN TIEMPO REAL")
    print("=" * 50)
    print("Controles:")
    print("  'q' = Salir")
    print("  'c' = Limpiar oración")
    print(f"  Acumulando {config.LONGITUD_SECUENCIA} frames antes de predecir...")
    print("=" * 50)
    
    etiqueta_prediccion = "Acumulando..."
    confianza = 0.0
    oracion = ""
    hay_mano = False
    
    while True:
        tiempo_inicio = time.time()
        ret, frame = captura.read()
        if not ret:
            break
        
        h, w, c = frame.shape
        
        # Procesar frame con MediaPipe Holistic
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        resultados = holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        # Verificar si hay manos detectadas
        hay_mano = (resultados.left_hand_landmarks is not None or 
                    resultados.right_hand_landmarks is not None)
        
        # Extraer y normalizar landmarks del frame actual
        landmarks_frame = extraer_landmarks_frame(resultados)
        landmarks_normalizado = normalizar_frame(landmarks_frame, buffer_secuencia)
        
        # Agregar al buffer
        buffer_secuencia.append(landmarks_normalizado)
        
        # Dibujar landmarks sobre el frame
        dibujar_landmarks(frame, resultados, mp_holistic, mp_dibujo)
        
        # Predecir solo cuando el buffer está lleno Y hay mano detectada
        if len(buffer_secuencia) == config.LONGITUD_SECUENCIA and hay_mano:
            # Preparar secuencia para el modelo
            secuencia = np.array(list(buffer_secuencia))  # (30, 225)
            
            if detalles_entrada[0]['dtype'] == np.float32:
                datos_entrada = np.float32(secuencia)
            else:
                datos_entrada = secuencia
            
            datos_entrada = np.expand_dims(datos_entrada, axis=0)  # (1, 30, 225)
            
            # Inferencia
            interprete.set_tensor(detalles_entrada[0]['index'], datos_entrada)
            interprete.invoke()
            datos_salida = interprete.get_tensor(detalles_salida[0]['index'])
            
            indice_prediccion = np.argmax(datos_salida)
            etiqueta_prediccion = nombres_clases[indice_prediccion]
            confianza = datos_salida[0][indice_prediccion]
            
            # Umbral de confianza
            if confianza < 0.6:
                etiqueta_prediccion = "Incierto"
            
            # Actualizar oración
            oracion = procesador.procesar_prediccion(etiqueta_prediccion)
            
        elif len(buffer_secuencia) < config.LONGITUD_SECUENCIA:
            etiqueta_prediccion = f"Acumulando... ({len(buffer_secuencia)}/{config.LONGITUD_SECUENCIA})"
            confianza = 0.0
        elif not hay_mano:
            etiqueta_prediccion = "Sin manos"
            oracion = procesador.obtener_oracion()
        
        # Cálculo de FPS
        fps = 1.0 / (time.time() - tiempo_inicio + 0.0001)
        
        # === VISUALIZACIÓN ===
        
        # Barra de estado superior
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)  # Fondo negro
        
        # Predicción
        color_pred = (0, 255, 0) if confianza >= 0.6 else (0, 165, 255)
        cv2.putText(frame, f"Seña: {etiqueta_prediccion}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_pred, 2)
        
        # Confianza y FPS
        cv2.putText(frame, f"Confianza: {confianza:.0%}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps:.0f}", 
                    (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Indicador de mano
        estado_mano = "MANO DETECTADA" if hay_mano else "SIN MANO"
        color_mano = (0, 255, 0) if hay_mano else (0, 0, 255)
        cv2.putText(frame, estado_mano, 
                    (w - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_mano, 1)
        
        # Barra de oración inferior
        cv2.rectangle(frame, (0, h - 50), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, f"Oracion: {oracion}", 
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Barra de progreso del buffer
        progreso = len(buffer_secuencia) / config.LONGITUD_SECUENCIA
        ancho_barra = int(w * progreso)
        color_barra = (0, 255, 0) if progreso >= 1.0 else (0, 165, 255)
        cv2.rectangle(frame, (0, h - 55), (ancho_barra, h - 50), color_barra, -1)
        
        cv2.imshow('Reconocimiento de Señas (LSTM + Landmarks)', frame)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('c'):
            procesador.limpiar()
            buffer_secuencia.clear()
            oracion = ""
            etiqueta_prediccion = "Acumulando..."
            confianza = 0.0
    
    captura.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Inferencia finalizada.")


if __name__ == "__main__":
    ejecutar_inferencia()
