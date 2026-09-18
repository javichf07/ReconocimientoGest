"""
Módulo 2: Preprocesamiento y Normalización Espacial
=====================================================
Implementa tres sub-módulos:

  2.1 Normalización Espacial (Invarianza):
      - Traslada la pose al centro del tórax
      - Traslada cada mano a su muñeca (landmark 0)
      - Escala todas las coordenadas por la distancia entre hombros

  2.2 Ingeniería de Características:
      - Distancias euclidianas entre yemas de los dedos (10 por mano)
      - Ángulos articulares entre falanges (15 por mano)

  2.3 Selección Estocástica de Frames:
      - Elimina frames estáticos (holds) durante entrenamiento

Uso:
    preprocesador = PreprocesadorCSLR()
    tensor = preprocesador.procesar_frame(pose_lm, mano_izq_lm, mano_der_lm)
    secuencia_filtrada = preprocesador.seleccionar_frames(secuencia, entrenamiento=True)
"""

import numpy as np
from itertools import combinations
from . import config_cslr as cfg


# ============================================================
# 2.1 NORMALIZACIÓN ESPACIAL
# ============================================================

def calcular_centro_torax(pose_landmarks):
    """
    Calcula el centro del tórax como el punto medio entre los hombros.
    
    Args:
        pose_landmarks: np.array (33, 3) de landmarks de pose
        
    Returns:
        np.array (3,) con las coordenadas del centro del tórax
    """
    hombro_izq = pose_landmarks[cfg.IDX_HOMBRO_IZQUIERDO]
    hombro_der = pose_landmarks[cfg.IDX_HOMBRO_DERECHO]
    return (hombro_izq + hombro_der) / 2.0


def calcular_distancia_hombros(pose_landmarks):
    """
    Calcula la distancia euclidiana entre los hombros.
    Se usa como factor de escala para invarianza de tamaño.
    
    Args:
        pose_landmarks: np.array (33, 3) de landmarks de pose
        
    Returns:
        float: distancia euclidiana entre hombros (mínimo 1e-6)
    """
    hombro_izq = pose_landmarks[cfg.IDX_HOMBRO_IZQUIERDO]
    hombro_der = pose_landmarks[cfg.IDX_HOMBRO_DERECHO]
    distancia = np.linalg.norm(hombro_izq - hombro_der)
    return max(distancia, 1e-6)  # Evitar división por cero


def normalizar_pose(pose_landmarks):
    """
    Normaliza los landmarks de pose para invarianza espacial:
    1. Traslada el origen al centro del tórax
    2. Escala por la distancia entre hombros
    
    Args:
        pose_landmarks: np.array (33, 3) de landmarks de pose
        
    Returns:
        tuple: (pose_normalizada, centro_torax, distancia_hombros)
            - pose_normalizada: np.array (33, 3) normalizado
            - centro_torax: np.array (3,) punto de referencia usado
            - distancia_hombros: float factor de escala usado
    """
    centro = calcular_centro_torax(pose_landmarks)
    escala = calcular_distancia_hombros(pose_landmarks)
    
    pose_norm = (pose_landmarks - centro) / escala
    
    return pose_norm, centro, escala


def normalizar_mano(mano_landmarks, distancia_hombros):
    """
    Normaliza los landmarks de una mano para invarianza espacial:
    1. Traslada el origen a la muñeca (landmark 0)
    2. Escala por la distancia entre hombros (misma escala que pose)
    
    Usar la distancia entre hombros como escala (en vez de un tamaño
    de la mano) garantiza que la relación espacial mano-cuerpo se preserve.
    
    Args:
        mano_landmarks: np.array (21, 3) de landmarks de mano
        distancia_hombros: float factor de escala (de la pose)
        
    Returns:
        np.array (21, 3) normalizado (origen en muñeca, escala por hombros)
    """
    muneca = mano_landmarks[cfg.IDX_MUNECA]
    mano_norm = (mano_landmarks - muneca) / distancia_hombros
    return mano_norm


# ============================================================
# 2.2 INGENIERÍA DE CARACTERÍSTICAS
# ============================================================

def calcular_distancias_yemas(mano_landmarks_3d):
    """
    Calcula las distancias euclidianas entre todas las combinaciones
    de yemas de los dedos. C(5,2) = 10 pares.
    
    Pares: (Pulgar-Índice), (Pulgar-Medio), (Pulgar-Anular), (Pulgar-Meñique),
           (Índice-Medio), (Índice-Anular), (Índice-Meñique),
           (Medio-Anular), (Medio-Meñique), (Anular-Meñique)
    
    Args:
        mano_landmarks_3d: np.array (21, 3) landmarks de mano (normalizados o no)
        
    Returns:
        np.array (10,) con las 10 distancias euclidianas entre yemas
    """
    yemas = mano_landmarks_3d[cfg.INDICES_YEMAS]  # (5, 3)
    
    distancias = []
    for i, j in combinations(range(len(yemas)), 2):
        dist = np.linalg.norm(yemas[i] - yemas[j])
        distancias.append(dist)
    
    return np.array(distancias, dtype=np.float32)  # (10,)


def calcular_angulo_articular(punto_padre, punto_articulacion, punto_hijo):
    """
    Calcula el ángulo (en radianes) en una articulación donde se unen dos huesos.
    
    El ángulo se calcula entre los vectores:
      v1 = padre → articulación
      v2 = hijo → articulación
      
    Usando: θ = arccos(v1 · v2 / (|v1| × |v2|))
    
    Args:
        punto_padre: np.array (3,) coordenadas del punto padre
        punto_articulacion: np.array (3,) coordenadas de la articulación
        punto_hijo: np.array (3,) coordenadas del punto hijo
        
    Returns:
        float: ángulo en radianes [0, π]
    """
    v1 = punto_padre - punto_articulacion
    v2 = punto_hijo - punto_articulacion
    
    norma_v1 = np.linalg.norm(v1)
    norma_v2 = np.linalg.norm(v2)
    
    if norma_v1 < 1e-8 or norma_v2 < 1e-8:
        return 0.0  # Puntos coincidentes, ángulo indefinido
    
    # Producto escalar normalizado, clamp a [-1, 1] para evitar errores numéricos
    cos_angulo = np.clip(np.dot(v1, v2) / (norma_v1 * norma_v2), -1.0, 1.0)
    
    return float(np.arccos(cos_angulo))


def calcular_angulos_falanges(mano_landmarks_3d):
    """
    Calcula la matriz de ángulos articulares entre todas las
    conexiones de falanges de la mano.
    
    Para cada articulación definida en cfg.ARTICULACIONES_ANGULOS,
    calcula el ángulo entre los dos huesos que se unen en ese punto.
    
    5 dedos × 3 ángulos por dedo = 15 ángulos totales.
    
    Args:
        mano_landmarks_3d: np.array (21, 3) landmarks de mano
        
    Returns:
        np.array (15,) con los ángulos articulares en radianes [0, π]
    """
    angulos = []
    
    for padre, articulacion, hijo in cfg.ARTICULACIONES_ANGULOS:
        angulo = calcular_angulo_articular(
            mano_landmarks_3d[padre],
            mano_landmarks_3d[articulacion],
            mano_landmarks_3d[hijo]
        )
        angulos.append(angulo)
    
    return np.array(angulos, dtype=np.float32)  # (15,)


def construir_features_mano(mano_landmarks, distancia_hombros):
    """
    Construye el vector completo de características de una mano:
      - Coordenadas normalizadas: 21 × 3 = 63 valores
      - Distancias entre yemas: 10 valores
      - Ángulos de falanges: 15 valores
      - Total: 88 valores
    
    Args:
        mano_landmarks: np.array (21, 3) landmarks crudos
        distancia_hombros: float factor de escala
        
    Returns:
        np.array (88,) vector de features de la mano
    """
    # Normalizar coordenadas (origen en muñeca, escala por hombros)
    mano_norm = normalizar_mano(mano_landmarks, distancia_hombros)
    
    # Calcular features adicionales sobre los landmarks normalizados
    distancias = calcular_distancias_yemas(mano_norm)
    angulos = calcular_angulos_falanges(mano_norm)
    
    # Concatenar: coords(63) + distancias(10) + ángulos(15) = 88
    return np.concatenate([
        mano_norm.flatten(),  # (63,)
        distancias,           # (10,)
        angulos               # (15,)
    ])


# ============================================================
# 2.3 SELECCIÓN ESTOCÁSTICA DE FRAMES
# ============================================================

def calcular_energia_cinetica(frame_actual, frame_anterior):
    """
    Calcula la energía cinética entre dos frames consecutivos
    como la suma de los cuadrados de las diferencias (velocidad²).
    
    Un valor bajo indica que el usuario mantiene una postura estática (hold).
    Un valor alto indica movimiento activo.
    
    Args:
        frame_actual: np.array (D,) vector de features del frame actual
        frame_anterior: np.array (D,) vector de features del frame anterior
        
    Returns:
        float: energía cinética (≥ 0)
    """
    delta = frame_actual - frame_anterior
    return float(np.sum(delta ** 2))


def seleccionar_frames_estocastico(secuencia, entrenamiento=True, 
                                    umbral=cfg.UMBRAL_ENERGIA_CINETICA,
                                    prob_drop=cfg.PROB_ELIMINACION_ESTATICOS):
    """
    Implementa la Selección Estocástica de Frames para eliminar
    frames donde el usuario mantiene una postura estática (holds).
    
    Solo se aplica durante entrenamiento. En inferencia, todos los
    frames se procesan para mantener la continuidad temporal.
    
    Algoritmo:
    1. Para cada par de frames consecutivos, calcula la energía cinética
    2. Si E_k < umbral → frame estático (hold)
    3. Con probabilidad prob_drop, elimina el frame estático
    4. Siempre conserva al menos 1/3 de la secuencia original
    
    Args:
        secuencia: np.array (T, D) secuencia de features
        entrenamiento: bool, si True aplica selección; si False retorna sin cambios
        umbral: float, umbral de energía cinética mínima
        prob_drop: float [0,1], probabilidad de eliminar un frame estático
        
    Returns:
        np.array (T', D) secuencia filtrada (T' ≤ T)
    """
    if not entrenamiento or len(secuencia) < 3:
        return secuencia
    
    indices_a_mantener = [0]  # Siempre conservar el primer frame
    
    for i in range(1, len(secuencia)):
        energia = calcular_energia_cinetica(secuencia[i], secuencia[i - 1])
        
        if energia >= umbral:
            # Frame dinámico: siempre conservar
            indices_a_mantener.append(i)
        else:
            # Frame estático: eliminar con probabilidad prob_drop
            if np.random.random() > prob_drop:
                indices_a_mantener.append(i)
    
    # Garantizar mínimo 1/3 de la secuencia original
    minimo_frames = max(len(secuencia) // 3, 2)
    if len(indices_a_mantener) < minimo_frames:
        # Restaurar frames distribuidos uniformemente
        indices_a_mantener = list(np.linspace(0, len(secuencia) - 1, 
                                              minimo_frames, dtype=int))
    
    return secuencia[indices_a_mantener]


# ============================================================
# PREPROCESADOR COMPLETO (ORQUESTADOR)
# ============================================================

class PreprocesadorCSLR:
    """
    Clase principal que orquesta todo el preprocesamiento:
    normalización espacial, ingeniería de features y selección de frames.
    
    Genera un tensor de 275 dimensiones por frame:
      - Pose normalizada: 99 valores (33 × 3)
      - Mano izquierda: 88 valores (63 coords + 10 distancias + 15 ángulos)
      - Mano derecha: 88 valores
    
    Atributos:
        ultimo_frame: almacena el último frame procesado para cálculo de flujo cinético
    """
    
    def __init__(self):
        """Inicializa el preprocesador."""
        self.ultimo_frame = None
    
    def procesar_frame(self, pose_landmarks, mano_izq_landmarks, mano_der_landmarks):
        """
        Procesa los landmarks crudos de un frame y genera el tensor de features.
        
        Args:
            pose_landmarks: np.array (33, 3) o None
            mano_izq_landmarks: np.array (21, 3) o None
            mano_der_landmarks: np.array (21, 3) o None
            
        Returns:
            np.array (275,) tensor de features del frame.
            Componentes con landmarks no detectados se rellenan con ceros.
        """
        # === Pose ===
        if pose_landmarks is not None:
            pose_norm, centro, distancia_hombros = normalizar_pose(pose_landmarks)
            features_pose = pose_norm.flatten()  # (99,)
        else:
            features_pose = np.zeros(cfg.DIM_POSE, dtype=np.float32)
            distancia_hombros = 1.0  # Fallback si no hay pose
        
        # === Mano Izquierda ===
        if mano_izq_landmarks is not None:
            features_mano_izq = construir_features_mano(mano_izq_landmarks, distancia_hombros)
        else:
            features_mano_izq = np.zeros(cfg.DIM_MANO_TOTAL, dtype=np.float32)
        
        # === Mano Derecha ===
        if mano_der_landmarks is not None:
            features_mano_der = construir_features_mano(mano_der_landmarks, distancia_hombros)
        else:
            features_mano_der = np.zeros(cfg.DIM_MANO_TOTAL, dtype=np.float32)
        
        # === Concatenar: Pose(99) + ManoIzq(88) + ManoDer(88) = 275 ===
        tensor = np.concatenate([
            features_pose,      # (99,)
            features_mano_izq,  # (88,)
            features_mano_der   # (88,)
        ]).astype(np.float32)
        
        assert tensor.shape[0] == cfg.DIM_FEATURE_FRAME, \
            f"Dimensión inesperada: {tensor.shape[0]} != {cfg.DIM_FEATURE_FRAME}"
        
        return tensor
    
    def procesar_secuencia_desde_video(self, frames_landmarks, entrenamiento=True):
        """
        Procesa una secuencia completa de landmarks (de un video) y genera
        el tensor de features con selección estocástica de frames.
        
        Args:
            frames_landmarks: lista de dicts, cada uno con:
                'pose': np.array (33, 3) o None
                'mano_izq': np.array (21, 3) o None
                'mano_der': np.array (21, 3) o None
            entrenamiento: bool, si True aplica selección estocástica
            
        Returns:
            np.array (T, 275) secuencia de features (posiblemente filtrada)
        """
        # Procesar cada frame
        tensores = []
        for frame_lm in frames_landmarks:
            tensor = self.procesar_frame(
                frame_lm.get('pose'),
                frame_lm.get('mano_izq'),
                frame_lm.get('mano_der')
            )
            tensores.append(tensor)
        
        if len(tensores) == 0:
            return np.zeros((0, cfg.DIM_FEATURE_FRAME), dtype=np.float32)
        
        secuencia = np.array(tensores)  # (T, 275)
        
        # Aplicar selección estocástica de frames
        secuencia = seleccionar_frames_estocastico(secuencia, entrenamiento=entrenamiento)
        
        return secuencia
    
    def calcular_flujo_cinetico(self, secuencia):
        """
        Calcula el flujo cinético (velocidad) como la diferencia
        vectorial entre frames consecutivos: Δt = frame_t - frame_{t-1}
        
        El primer frame tiene flujo cero (no tiene frame anterior).
        
        Args:
            secuencia: np.array (T, 275) secuencia de features
            
        Returns:
            np.array (T, 275) flujo cinético (misma dimensión que la entrada)
        """
        flujo = np.zeros_like(secuencia)
        if len(secuencia) > 1:
            flujo[1:] = secuencia[1:] - secuencia[:-1]
        return flujo
    
    def separar_flujos(self, secuencia):
        """
        Separa una secuencia de features en los tres flujos del encoder:
          1. Pose general (99 dims)
          2. Semántica de manos (176 dims: 88 + 88)
          3. Flujo cinético (275 dims: delta temporal de todo)
        
        Args:
            secuencia: np.array (T, 275) secuencia de features
            
        Returns:
            dict con:
                'pose': np.array (T, 99)
                'manos': np.array (T, 176)
                'cinetico': np.array (T, 275)
        """
        flujo_pose = secuencia[:, :cfg.DIM_FLUJO_POSE]          # (T, 99)
        flujo_manos = secuencia[:, cfg.DIM_FLUJO_POSE:]          # (T, 176)
        flujo_cinetico = self.calcular_flujo_cinetico(secuencia)  # (T, 275)
        
        return {
            'pose': flujo_pose,
            'manos': flujo_manos,
            'cinetico': flujo_cinetico
        }
    
    def resetear(self):
        """Resetea el estado interno del preprocesador."""
        self.ultimo_frame = None
