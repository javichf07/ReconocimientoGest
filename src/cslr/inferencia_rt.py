"""
Módulo 4: Inferencia en Tiempo Real (FIFO + Sliding Window)
=============================================================
Implementa una clase manejadora de inferencia que:
  - Mantiene una cola FIFO circular de ~90 frames continuos
  - Cada 10-15 frames, extrae un sub-bloque y lo inyecta al modelo
  - Genera predicciones textuales parciales al vuelo

Uso:
    python -m src.cslr.inferencia_rt
"""

import cv2
import numpy as np
import time
import os
import tensorflow as tf
from collections import deque
from . import config_cslr as cfg
from .extraccion import ExtractorLandmarks
from .preprocesamiento import PreprocesadorCSLR


# Umbral mínimo de confianza para aceptar una predicción
UMBRAL_CONFIANZA = 0.30


class ManejadorInferencia:
    """
    Manejador de inferencia en tiempo real con cola FIFO y ventana deslizante.
    """
    
    def __init__(self, ruta_modelo_tflite, ruta_labels):
        # Buffer FIFO circular
        self.buffer = deque(maxlen=cfg.TAMANO_BUFFER_FIFO)
        self.contador_frames = 0
        self.prediccion_actual = "Esperando..."
        self.confianza_actual = 0.0
        self.probabilidades = None  # Array de probabilidades por clase
        
        # Cargar modelo TFLite
        print("Cargando modelo CSLR TFLite...")
        self.interprete = tf.lite.Interpreter(model_path=ruta_modelo_tflite)
        self.interprete.allocate_tensors()
        self.detalles_entrada = self.interprete.get_input_details()
        self.detalles_salida = self.interprete.get_output_details()
        
        # Cargar etiquetas
        with open(ruta_labels, 'r', encoding='utf-8') as f:
            self.nombres_clases = [line.strip() for line in f.readlines() if line.strip()]
        
        # Mapear entradas por FORMA (mucho más robusto que por nombre)
        # Las formas esperadas son:
        #   pose:     (1, T, 99)
        #   manos:    (1, T, 176)
        #   cinético: (1, T, 275)
        self.mapa_entradas = {}
        for det in self.detalles_entrada:
            dim_features = det['shape'][-1]  # Última dimensión = número de features
            if dim_features == cfg.DIM_FLUJO_POSE:       # 99
                self.mapa_entradas['pose'] = det['index']
            elif dim_features == cfg.DIM_FLUJO_MANOS:     # 176
                self.mapa_entradas['manos'] = det['index']
            elif dim_features == cfg.DIM_FLUJO_CINETICO:  # 275
                self.mapa_entradas['cinetico'] = det['index']
        
        print(f"Modelo cargado. Clases: {self.nombres_clases}")
        print(f"Entradas mapeadas por forma: {self.mapa_entradas}")
        for det in self.detalles_entrada:
            print(f"  '{det['name']}' → shape={det['shape']} → "
                  f"features_dim={det['shape'][-1]}")
        
        # Verificar que se mapearon las 3 entradas
        for flujo in ['pose', 'manos', 'cinetico']:
            if flujo not in self.mapa_entradas:
                print(f"  ⚠ ALERTA: No se pudo mapear la entrada '{flujo}'!")
    
    def agregar_frame(self, features_frame):
        """Agrega un frame al buffer. Retorna True si se hizo una predicción."""
        self.buffer.append(features_frame)
        self.contador_frames += 1
        
        if (self.contador_frames >= cfg.PASO_DESLIZAMIENTO and 
            len(self.buffer) >= cfg.LONGITUD_SECUENCIA_CSLR):
            
            self.contador_frames = 0
            self._predecir()
            return True
        
        return False
    
    def _predecir(self):
        """Extrae sub-secuencia del buffer, la pasa al modelo y actualiza predicción."""
        # Extraer los últimos LONGITUD_SECUENCIA_CSLR frames
        buffer_list = list(self.buffer)
        secuencia = np.array(
            buffer_list[-cfg.LONGITUD_SECUENCIA_CSLR:], 
            dtype=np.float32
        )  # (60, 275)
        
        # Separar en los 3 flujos
        pose = secuencia[:, :cfg.DIM_FLUJO_POSE]           # (60, 99)
        manos = secuencia[:, cfg.DIM_FLUJO_POSE:]           # (60, 176)
        
        # Flujo cinético (velocidad temporal)
        cinetico = np.zeros_like(secuencia)                  # (60, 275)
        cinetico[1:] = secuencia[1:] - secuencia[:-1]
        
        # Asignar entradas al modelo por FORMA (no por nombre)
        try:
            self.interprete.set_tensor(
                self.mapa_entradas['pose'],
                np.expand_dims(pose, 0).astype(np.float32)      # (1, 60, 99)
            )
            self.interprete.set_tensor(
                self.mapa_entradas['manos'],
                np.expand_dims(manos, 0).astype(np.float32)     # (1, 60, 176)
            )
            self.interprete.set_tensor(
                self.mapa_entradas['cinetico'],
                np.expand_dims(cinetico, 0).astype(np.float32)  # (1, 60, 275)
            )
            
            self.interprete.invoke()
            salida = self.interprete.get_tensor(self.detalles_salida[0]['index'])
            
            self.probabilidades = salida[0]  # Array de probabilidades por clase
            idx = np.argmax(self.probabilidades)
            self.confianza_actual = float(self.probabilidades[idx])
            
            if self.confianza_actual >= UMBRAL_CONFIANZA:
                self.prediccion_actual = self.nombres_clases[idx]
            else:
                self.prediccion_actual = "---"
                
        except Exception as e:
            print(f"Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            self.prediccion_actual = "Error"
            self.confianza_actual = 0.0
            self.probabilidades = None
    
    def obtener_prediccion(self):
        return self.prediccion_actual, self.confianza_actual
    
    def obtener_progreso_buffer(self):
        return len(self.buffer) / cfg.TAMANO_BUFFER_FIFO


def ejecutar_inferencia():
    """Función principal de inferencia en tiempo real."""
    
    # Verificar archivos
    if not os.path.exists(cfg.RUTA_MODELO_CSLR_TFLITE):
        print(f"Modelo no encontrado: {cfg.RUTA_MODELO_CSLR_TFLITE}")
        print("Ejecuta: python -m src.cslr.exportar_int8")
        return
    
    if not os.path.exists(cfg.RUTA_LABELS_CSLR):
        print(f"Labels no encontradas: {cfg.RUTA_LABELS_CSLR}")
        return
    
    # Inicializar módulos
    print("=" * 60)
    print("INFERENCIA CSLR EN TIEMPO REAL")
    print("=" * 60)
    
    extractor = ExtractorLandmarks()
    preprocesador = PreprocesadorCSLR()
    manejador = ManejadorInferencia(cfg.RUTA_MODELO_CSLR_TFLITE, cfg.RUTA_LABELS_CSLR)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return
    
    print(f"\nBuffer FIFO: {cfg.TAMANO_BUFFER_FIFO} frames")
    print(f"Secuencia: {cfg.LONGITUD_SECUENCIA_CSLR} frames")
    print(f"Predicción cada: {cfg.PASO_DESLIZAMIENTO} frames")
    print(f"Umbral confianza: {UMBRAL_CONFIANZA:.0%}")
    print("Controles: 'q'=salir, 'c'=limpiar")
    print("=" * 60)
    
    oracion = ""
    ultima_prediccion = ""
    contador_repeticiones = 0
    
    # Colores por clase (generados automáticamente)
    colores_clase = [
        (0, 255, 0),     # Verde
        (255, 165, 0),   # Naranja (BGR)
        (255, 0, 0),     # Azul (BGR)
        (0, 255, 255),   # Amarillo (BGR)
        (255, 0, 255),   # Magenta
        (0, 165, 255),   # Naranja claro
        (255, 255, 0),   # Cyan
        (128, 0, 255),   # Violeta
    ]
    
    while True:
        t_inicio = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w, _ = frame.shape
        
        # 1. Extraer landmarks
        resultados = extractor.procesar_frame(frame)
        mano_izq, mano_der = extractor.separar_manos(resultados['manos'])
        
        # 2. Preprocesar → tensor 275D
        features = preprocesador.procesar_frame(
            resultados['pose'], mano_izq, mano_der
        )
        
        # 3. Agregar al buffer FIFO + predecir si corresponde
        predijo = manejador.agregar_frame(features)
        prediccion, confianza = manejador.obtener_prediccion()
        
        # 4. Construir oración (agregar si la seña se mantiene 3+ predicciones)
        if predijo:
            if prediccion == ultima_prediccion and prediccion not in ("---", "Error", "Esperando..."):
                contador_repeticiones += 1
                if contador_repeticiones == 3:  # Confirmar seña después de 3 predicciones consecutivas
                    if not oracion.endswith(prediccion):
                        oracion = (oracion + " " + prediccion).strip()
            else:
                contador_repeticiones = 0
            ultima_prediccion = prediccion
            
            # Debug: imprimir en consola cada predicción
            if manejador.probabilidades is not None:
                probs_str = " | ".join([
                    f"{manejador.nombres_clases[i]}:{manejador.probabilidades[i]:.0%}"
                    for i in range(len(manejador.nombres_clases))
                ])
                print(f"[{prediccion} {confianza:.0%}] {probs_str}")
        
        # 5. Dibujar esqueleto
        extractor.dibujar_en_frame(frame, resultados)
        
        # 6. VISUALIZACIÓN
        fps = 1.0 / (time.time() - t_inicio + 1e-6)
        hay_mano = mano_izq is not None or mano_der is not None
        
        # === Panel superior: predicción ===
        alto_panel = 40
        cv2.rectangle(frame, (0, 0), (w, alto_panel), (0, 0, 0), -1)
        
        color_pred = (0, 255, 0) if confianza >= 0.5 else (0, 200, 255) if confianza >= 0.3 else (0, 0, 255)
        cv2.putText(frame, f"{prediccion} ({confianza:.0%})", 
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_pred, 2)
        
        estado = "MANO OK" if hay_mano else "SIN MANO"
        color_estado = (0, 255, 0) if hay_mano else (0, 0, 255)
        cv2.putText(frame, f"FPS:{fps:.0f} | Buffer:{len(manejador.buffer)}/{cfg.TAMANO_BUFFER_FIFO} | {estado}",
                    (w - 380, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_estado, 1)
        
        # === Panel derecho: barras de probabilidad por clase ===
        if manejador.probabilidades is not None:
            num_clases = len(manejador.nombres_clases)
            barra_alto = 25
            barra_ancho_max = 200
            margen_x = w - barra_ancho_max - 90
            inicio_y = alto_panel + 10
            
            for i in range(num_clases):
                y = inicio_y + i * (barra_alto + 5)
                prob = float(manejador.probabilidades[i])
                barra_w = int(prob * barra_ancho_max)
                color = colores_clase[i % len(colores_clase)]
                
                # Fondo
                cv2.rectangle(frame, (margen_x - 5, y - 2), 
                             (margen_x + barra_ancho_max + 80, y + barra_alto + 2),
                             (30, 30, 30), -1)
                
                # Barra de probabilidad
                cv2.rectangle(frame, (margen_x + 60, y), 
                             (margen_x + 60 + barra_w, y + barra_alto),
                             color, -1)
                
                # Contorno de barra completa
                cv2.rectangle(frame, (margen_x + 60, y),
                             (margen_x + 60 + barra_ancho_max, y + barra_alto),
                             (100, 100, 100), 1)
                
                # Nombre de clase
                cv2.putText(frame, f"{manejador.nombres_clases[i]}",
                           (margen_x, y + barra_alto - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Porcentaje
                cv2.putText(frame, f"{prob:.0%}",
                           (margen_x + 65 + barra_ancho_max, y + barra_alto - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # === Panel inferior: oración acumulada ===
        cv2.rectangle(frame, (0, h - 45), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, f"Oracion: {oracion if oracion else '(haz una sena)'}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Barra de progreso del buffer
        progreso = manejador.obtener_progreso_buffer()
        barra_w = int(w * progreso)
        cv2.rectangle(frame, (0, h - 50), (barra_w, h - 45),
                     (0, 255, 0) if progreso >= 0.7 else (0, 165, 255), -1)
        
        cv2.imshow('CSLR - Inferencia en Tiempo Real', frame)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('c'):
            oracion = ""
            contador_repeticiones = 0
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.liberar()
    print("\nInferencia finalizada.")


if __name__ == "__main__":
    ejecutar_inferencia()
