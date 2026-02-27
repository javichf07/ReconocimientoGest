import cv2
import tensorflow as tf
import numpy as np
import time
import os
import mediapipe as mp # Importar MediaPipe
from . import config
from .realtime_processor import ProcesadorTiempoReal

def ejecutar_inferencia():
    # Cargar Modelo TFLite
    if not os.path.exists(config.RUTA_MODELO_TFLITE):
        print(f"Modelo no encontrado en {config.RUTA_MODELO_TFLITE}. Ejecuta export.py primero.")
        return

    print("Cargando Intérprete TFLite...")
    interprete = tf.lite.Interpreter(model_path=config.RUTA_MODELO_TFLITE)
    interprete.allocate_tensors()

    detalles_entrada = interprete.get_input_details()
    detalles_salida = interprete.get_output_details()
    
    # Cargar Etiquetas
    ruta_etiquetas = os.path.join(config.RUTA_MODELOS, 'labels.txt')
    if not os.path.exists(ruta_etiquetas):
        print("Archivo de etiquetas no encontrado. Ejecuta train.py primero.")
        return
        
    with open(ruta_etiquetas, 'r') as f:
        nombres_clases = [line.strip() for line in f.readlines()]
    
    print(f"Clases: {nombres_clases}")

    # Inicializar Procesador
    procesador = ProcesadorTiempoReal()

    print("Inicializando MediaPipe Hands...")
    print("Inicializando MediaPipe Hands...")
    try:
        mp_manos = mp.solutions.hands
        mp_dibujo = mp.solutions.drawing_utils
    except Exception as e:
        print(f"\nERROR CRÍTICO CON MEDIAPIPE: {e}")
        print("Esto es 100% causado por una versión incompatible de 'protobuf'.")
        print("SOLUCIÓN: Ejecuta este comando en tu terminal e intenta de nuevo:")
        print("   pip install \"protobuf<4\"")
        return
    
    # Configurar deteccion de manos
    # min_detection_confidence: Confianza mínima para detectar mano inicial
    # min_tracking_confidence: Confianza mínima para seguir la mano
    manos = mp_manos.Hands(
        max_num_hands=1, # Por ahora solo 1 mano para simplificar
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Abrir Webcam
    captura = cv2.VideoCapture(0)
    
    if not captura.isOpened():
        print("No se puede abrir la cámara")
        return

    print("Iniciando Inferencia Híbrida. Levanta tu mano para detectar.")
    print("Presiona 'q' para salir, 'c' para limpiar oración.")
    
    while True:
        tiempo_inicio = time.time()
        ret, frame = captura.read()
        if not ret:
            break
            
        h, w, c = frame.shape
        
        # 1. Detección de Manos con MediaPipe (Requiere RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = manos.process(frame_rgb)
        
        etiqueta_prediccion = "Esperando..."
        confianza = 0.0
        img_para_modelo = np.zeros((config.ALTO_IMG, config.ANCHO_IMG, 3), dtype=np.uint8)
        
        # Si se detectan manos...
        if resultados.multi_hand_landmarks:
            for landmarks_mano in resultados.multi_hand_landmarks:
                # 2. Dibujar Esqueleto (Siluetas)
                mp_dibujo.draw_landmarks(
                    frame, 
                    landmarks_mano, 
                    mp_manos.HAND_CONNECTIONS,
                    mp_dibujo.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_dibujo.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                
                # 3. Calcular Bounding Box (Caja alrededor de la mano) para Recorte Dinámico
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                
                for lm in landmarks_mano.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)

                # Añadir margen (padding) al recorte para que no quede muy apretado
                margen = 40
                x_min = max(0, x_min - margen)
                y_min = max(0, y_min - margen)
                x_max = min(w, x_max + margen)
                y_max = min(h, y_max + margen)
                
                # Dibujar la caja de recorte
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                
                # 4. Extraer la imagen de la mano (Recorte)
                mano_recortada = frame[y_min:y_max, x_min:x_max]
                
                if mano_recortada.size == 0:
                    continue

                # Preprocesamiento para MobileNet
                img_redimensionada = cv2.resize(mano_recortada, (config.ANCHO_IMG, config.ALTO_IMG))
                
                # Convertir a RGB para el modelo
                img_rgb_input = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2RGB)
                
                # Guardar para visualización debug
                img_para_modelo = img_redimensionada # Mostrar en BGR
                
                if detalles_entrada[0]['dtype'] == np.float32:
                    datos_entrada = (np.float32(img_rgb_input) / 127.5) - 1.0
                else:
                    datos_entrada = img_rgb_input

                datos_entrada = np.expand_dims(datos_entrada, axis=0)
                
                # 5. Inferencia
                interprete.set_tensor(detalles_entrada[0]['index'], datos_entrada)
                interprete.invoke()
                datos_salida = interprete.get_tensor(detalles_salida[0]['index'])
                
                indice_prediccion = np.argmax(datos_salida)
                etiqueta_prediccion = nombres_clases[indice_prediccion]
                confianza = datos_salida[0][indice_prediccion]
                
                # Umbral de confianza más estricto
                if confianza < 0.7:
                    etiqueta_prediccion = "Incierto"
                
                # Procesar oración solo si hay predicción válida
                oracion = procesador.procesar_prediccion(etiqueta_prediccion)
        else:
            # Si no hay mano, resetear o mantener espera
             etiqueta_prediccion = "No mano"
             oracion = procesador.obtener_oracion()
        
        # Cálculo de FPS
        fps = 1.0 / (time.time() - tiempo_inicio)
        
        # Visualización
        cv2.putText(frame, f"Pred: {etiqueta_prediccion} ({confianza:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Oracion: {oracion}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow('Reconocimiento de Lenguaje de Señas (Híbrido)', frame)
        cv2.imshow('Vista del Modelo', img_para_modelo)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('c'):
            procesador.limpiar()

    captura.release()
    cv2.destroyAllWindows()
    manos.close()

if __name__ == "__main__":
    ejecutar_inferencia()
