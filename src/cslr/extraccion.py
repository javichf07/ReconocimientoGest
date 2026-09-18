"""
Módulo 1: Extracción de Características
=========================================
Implementa instancias concurrentes y separadas para:
  - Pose Landmarker (rastreo del cuerpo, 33 puntos)
  - Hand Landmarker (rastreo de manos, 21 puntos × 2 manos)

NO utiliza MediaPipe Holistic ni Face Mesh para mantener
la eficiencia térmica y computacional en hardware móvil.

Uso:
    extractor = ExtractorLandmarks()
    landmarks = extractor.procesar_frame(frame_bgr)
    extractor.liberar()
"""

import numpy as np
import mediapipe as mp
from . import config_cslr as cfg


class ExtractorLandmarks:
    """
    Extrae landmarks de pose y manos de un frame de video
    usando instancias separadas de MediaPipe Pose y Hands.
    
    Atributos:
        pose: Instancia de mp.solutions.pose.Pose
        manos: Instancia de mp.solutions.hands.Hands
        mp_pose: Referencia al módulo de pose para constantes
        mp_hands: Referencia al módulo de hands para constantes
        mp_dibujo: Utilidades de dibujo de MediaPipe
    """
    
    def __init__(self):
        """Inicializa los rastreadores de pose y manos por separado."""
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_dibujo = mp.solutions.drawing_utils
        
        # Rastreador de Pose (cuerpo completo, sin face mesh)
        self.pose = self.mp_pose.Pose(
            model_complexity=cfg.COMPLEJIDAD_POSE,
            min_detection_confidence=cfg.CONFIANZA_DETECCION_POSE,
            min_tracking_confidence=cfg.CONFIANZA_RASTREO_POSE,
            enable_segmentation=False  # No necesitamos máscara de segmentación
        )
        
        # Rastreador de Manos (video continuo, modelo lite)
        self.manos = self.mp_hands.Hands(
            static_image_mode=False,       # Modo video continuo (tracking entre frames)
            max_num_hands=cfg.MAX_MANOS,    # Máximo 2 manos
            model_complexity=cfg.COMPLEJIDAD_MANOS,  # 0 = lite para móvil
            min_detection_confidence=cfg.CONFIANZA_DETECCION_MANOS,
            min_tracking_confidence=cfg.CONFIANZA_RASTREO_MANOS
        )
    
    def procesar_frame(self, frame_bgr):
        """
        Procesa un frame BGR de OpenCV y extrae landmarks de pose y manos.
        
        Args:
            frame_bgr: Frame BGR de OpenCV (numpy array HxWx3)
            
        Returns:
            dict con:
                'pose': np.array (33, 3) o None si no se detectó pose
                'manos': lista de dict con:
                    'landmarks': np.array (21, 3)
                    'lateralidad': 'Left' o 'Right'
                'resultados_pose': objeto MediaPipe para dibujo
                'resultados_manos': objeto MediaPipe para dibujo
        """
        import cv2
        
        # Convertir BGR → RGB (MediaPipe requiere RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # Optimización de rendimiento
        
        # === Procesar Pose ===
        resultado_pose = self.pose.process(frame_rgb)
        
        pose_landmarks = None
        if resultado_pose.pose_landmarks:
            pose_landmarks = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in resultado_pose.pose_landmarks.landmark
            ])  # Shape: (33, 3)
        
        # === Procesar Manos ===
        resultado_manos = self.manos.process(frame_rgb)
        
        manos_detectadas = []
        if resultado_manos.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(resultado_manos.multi_hand_landmarks):
                # Obtener lateralidad (Left/Right)
                lateralidad = 'Right'  # Default
                if resultado_manos.multi_handedness:
                    lateralidad = resultado_manos.multi_handedness[idx].classification[0].label
                
                landmarks_array = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in hand_landmarks.landmark
                ])  # Shape: (21, 3)
                
                manos_detectadas.append({
                    'landmarks': landmarks_array,
                    'lateralidad': lateralidad
                })
        
        return {
            'pose': pose_landmarks,
            'manos': manos_detectadas,
            'resultados_pose': resultado_pose,
            'resultados_manos': resultado_manos
        }
    
    def separar_manos(self, manos_detectadas):
        """
        Separa las manos detectadas en izquierda y derecha.
        Si una mano no se detectó, retorna None.
        
        Nota: MediaPipe reporta la lateralidad desde la perspectiva de la cámara
        (espejado). 'Right' de MediaPipe = mano izquierda del usuario.
        
        Args:
            manos_detectadas: lista de dicts de procesar_frame()['manos']
            
        Returns:
            (mano_izq, mano_der): arrays (21, 3) o None
        """
        mano_izq = None
        mano_der = None
        
        for mano in manos_detectadas:
            # MediaPipe 'Right' = mano izquierda del usuario (espejado por cámara)
            if mano['lateralidad'] == 'Right':
                mano_izq = mano['landmarks']
            elif mano['lateralidad'] == 'Left':
                mano_der = mano['landmarks']
        
        return mano_izq, mano_der
    
    def dibujar_en_frame(self, frame_bgr, resultados):
        """
        Dibuja los esqueletos de pose y manos sobre el frame.
        
        Args:
            frame_bgr: Frame BGR de OpenCV (se modifica in-place)
            resultados: dict retornado por procesar_frame()
        """
        # Dibujar Pose (esqueleto corporal, azul)
        if resultados['resultados_pose'].pose_landmarks:
            self.mp_dibujo.draw_landmarks(
                frame_bgr,
                resultados['resultados_pose'].pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_dibujo.DrawingSpec(color=(80, 110, 200), thickness=2, circle_radius=2),
                self.mp_dibujo.DrawingSpec(color=(80, 180, 200), thickness=2, circle_radius=1)
            )
        
        # Dibujar Manos
        if resultados['resultados_manos'].multi_hand_landmarks:
            for idx, hand_lms in enumerate(resultados['resultados_manos'].multi_hand_landmarks):
                # Verde para mano izquierda, Rojo para mano derecha
                lateralidad = 'Right'
                if resultados['resultados_manos'].multi_handedness:
                    lateralidad = resultados['resultados_manos'].multi_handedness[idx].classification[0].label
                
                if lateralidad == 'Right':  # Mano izquierda del usuario
                    color_punto = (0, 255, 0)
                    color_linea = (0, 200, 0)
                else:
                    color_punto = (0, 0, 255)
                    color_linea = (0, 0, 200)
                
                self.mp_dibujo.draw_landmarks(
                    frame_bgr,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_dibujo.DrawingSpec(color=color_punto, thickness=2, circle_radius=3),
                    self.mp_dibujo.DrawingSpec(color=color_linea, thickness=2, circle_radius=2)
                )
    
    def liberar(self):
        """Libera los recursos de MediaPipe."""
        self.pose.close()
        self.manos.close()
