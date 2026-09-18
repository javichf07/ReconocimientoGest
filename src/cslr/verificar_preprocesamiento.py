"""
Script de verificación del Módulo de Preprocesamiento CSLR.

Abre la webcam, extrae landmarks con los trackers separados (Pose + Hands),
los preprocesa con normalización espacial + features engineered, y
muestra en pantalla el esqueleto + información de depuración.

Uso:
    python -m src.cslr.verificar_preprocesamiento
"""

import cv2
import numpy as np
import time
from . import config_cslr as cfg
from .extraccion import ExtractorLandmarks
from .preprocesamiento import PreprocesadorCSLR


def ejecutar_verificacion():
    """Abre la webcam y verifica que la extracción + preprocesamiento funcionen."""
    
    print("=" * 60)
    print("VERIFICACIÓN: Extracción + Preprocesamiento CSLR")
    print("=" * 60)
    print(f"Dimensiones esperadas por frame: {cfg.DIM_FEATURE_FRAME}")
    print(f"  Pose: {cfg.DIM_POSE} | Mano: {cfg.DIM_MANO_TOTAL} × {cfg.MAX_MANOS}")
    print(f"  Distancias yemas: {cfg.DIM_DISTANCIAS_YEMAS}/mano")
    print(f"  Ángulos falanges: {cfg.DIM_ANGULOS_FALANGES}/mano")
    print("=" * 60)
    
    # Inicializar módulos
    print("\nInicializando Pose Tracker + Hand Tracker (separados)...")
    extractor = ExtractorLandmarks()
    preprocesador = PreprocesadorCSLR()
    
    # Abrir cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return
    
    print("Cámara abierta. Presiona 'q' para salir.\n")
    
    while True:
        t_inicio = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w, _ = frame.shape
        
        # 1. Extraer landmarks
        resultados = extractor.procesar_frame(frame)
        mano_izq, mano_der = extractor.separar_manos(resultados['manos'])
        
        # 2. Preprocesar
        tensor = preprocesador.procesar_frame(
            resultados['pose'], mano_izq, mano_der
        )
        
        # 3. Dibujar esqueleto
        extractor.dibujar_en_frame(frame, resultados)
        
        # 4. Info en pantalla
        fps = 1.0 / (time.time() - t_inicio + 1e-6)
        
        # Fondo negro para texto
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        
        # Estado de detección
        pose_ok = "✓" if resultados['pose'] is not None else "✗"
        mano_i_ok = "✓" if mano_izq is not None else "✗"
        mano_d_ok = "✓" if mano_der is not None else "✗"
        
        cv2.putText(frame, f"FPS: {fps:.0f} | Tensor: {tensor.shape}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Pose: {pose_ok} | Mano Izq: {mano_i_ok} | Mano Der: {mano_d_ok}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Mostrar primeros valores del tensor (depuración)
        # Pose (primeros 6 = 2 landmarks), Mano (ángulos)
        pose_vals = tensor[:6]
        cv2.putText(frame, f"Pose[0:6]: {np.array2string(pose_vals, precision=2, separator=',')}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 255), 1)
        
        # Ángulos de mano izquierda (si detectada)
        if mano_izq is not None:
            inicio_ang = cfg.DIM_POSE + cfg.DIM_MANO_COORDS + cfg.DIM_DISTANCIAS_YEMAS
            angulos = tensor[inicio_ang : inicio_ang + 5]  # Primeros 5 ángulos
            cv2.putText(frame, f"Angulos ManoIzq[0:5]: {np.array2string(angulos, precision=2)}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 150), 1)
        
        cv2.imshow('Verificacion CSLR - Preprocesamiento', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.liberar()
    print("Verificación completada.")


if __name__ == "__main__":
    ejecutar_verificacion()
