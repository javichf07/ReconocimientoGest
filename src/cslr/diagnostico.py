"""
Diagnóstico del pipeline CSLR.
Verifica cada componente para encontrar por qué no reconoce señas.

Uso: python -m src.cslr.diagnostico
"""

import os
import numpy as np
import tensorflow as tf
from . import config_cslr as cfg


def diagnostico():
    print("=" * 60)
    print("DIAGNÓSTICO CSLR")
    print("=" * 60)
    
    # 1. Verificar archivos
    print("\n--- 1. ARCHIVOS ---")
    archivos = {
        'Modelo TFLite': cfg.RUTA_MODELO_CSLR_TFLITE,
        'Modelo Keras': cfg.RUTA_MEJOR_MODELO_CSLR,
        'Labels': cfg.RUTA_LABELS_CSLR,
        'Datos CSLR': cfg.DIR_LANDMARKS_CSLR,
    }
    for nombre, ruta in archivos.items():
        existe = os.path.exists(ruta)
        tamano = ""
        if existe and os.path.isfile(ruta):
            tamano = f" ({os.path.getsize(ruta) / 1024:.1f} KB)"
        print(f"  {'✓' if existe else '✗'} {nombre}: {ruta}{tamano}")
    
    # 2. Verificar labels
    print("\n--- 2. LABELS ---")
    if os.path.exists(cfg.RUTA_LABELS_CSLR):
        with open(cfg.RUTA_LABELS_CSLR, 'r', encoding='utf-8') as f:
            labels = [l.strip() for l in f.readlines() if l.strip()]
        print(f"  Clases ({len(labels)}): {labels}")
    else:
        print("  ✗ No hay labels!")
        return
    
    # 3. Verificar datos de entrenamiento
    print("\n--- 3. DATOS DE ENTRENAMIENTO ---")
    if os.path.exists(cfg.DIR_LANDMARKS_CSLR):
        for clase in sorted(os.listdir(cfg.DIR_LANDMARKS_CSLR)):
            ruta = os.path.join(cfg.DIR_LANDMARKS_CSLR, clase)
            if os.path.isdir(ruta):
                npys = [f for f in os.listdir(ruta) if f.endswith('.npy')]
                print(f"  Clase '{clase}': {len(npys)} secuencias")
                if npys:
                    muestra = np.load(os.path.join(ruta, npys[0]))
                    print(f"    Forma: {muestra.shape}, dtype: {muestra.dtype}")
                    print(f"    Rango valores: [{muestra.min():.4f}, {muestra.max():.4f}]")
                    print(f"    Media: {muestra.mean():.4f}, Std: {muestra.std():.4f}")
                    # Verificar si hay muchos ceros (manos no detectadas)
                    pct_ceros = (muestra == 0).sum() / muestra.size * 100
                    print(f"    Porcentaje de ceros: {pct_ceros:.1f}%")
    
    # 4. Verificar modelo TFLite
    print("\n--- 4. MODELO TFLITE ---")
    if os.path.exists(cfg.RUTA_MODELO_CSLR_TFLITE):
        interprete = tf.lite.Interpreter(model_path=cfg.RUTA_MODELO_CSLR_TFLITE)
        interprete.allocate_tensors()
        
        entradas = interprete.get_input_details()
        salidas = interprete.get_output_details()
        
        print(f"  Número de entradas: {len(entradas)}")
        for i, det in enumerate(entradas):
            print(f"    Entrada {i}: name='{det['name']}', shape={det['shape']}, dtype={det['dtype']}")
        
        print(f"  Número de salidas: {len(salidas)}")
        for i, det in enumerate(salidas):
            print(f"    Salida {i}: name='{det['name']}', shape={det['shape']}, dtype={det['dtype']}")
        
        # 5. Hacer una predicción de prueba
        print("\n--- 5. PREDICCIÓN DE PRUEBA ---")
        
        if len(entradas) == 3:
            print("  Modelo tiene 3 entradas (pose, manos, cinético)")
            # Intentar mapear por nombre
            for det in entradas:
                nombre = det['name'].lower()
                shape = det['shape']
                dummy = np.random.randn(*shape).astype(np.float32) * 0.1
                interprete.set_tensor(det['index'], dummy)
                print(f"    Entrada '{det['name']}': shape {shape} → OK")
        elif len(entradas) == 1:
            print("  Modelo tiene 1 entrada (concatenada)")
            shape = entradas[0]['shape']
            dummy = np.random.randn(*shape).astype(np.float32) * 0.1
            interprete.set_tensor(entradas[0]['index'], dummy)
            print(f"    Entrada '{entradas[0]['name']}': shape {shape} → OK")
        
        interprete.invoke()
        resultado = interprete.get_tensor(salidas[0]['index'])
        print(f"\n  Salida raw (datos random): {resultado}")
        print(f"  Forma salida: {resultado.shape}")
        print(f"  Suma probabilidades: {resultado.sum():.4f} (debería ser ~1.0)")
        print(f"  Max prob: {resultado.max():.4f}, Clase predicha: {np.argmax(resultado)}")
        
        # 6. Predicción con datos reales (si hay)
        print("\n--- 6. PREDICCIÓN CON DATOS REALES ---")
        if os.path.exists(cfg.DIR_LANDMARKS_CSLR):
            for clase in sorted(os.listdir(cfg.DIR_LANDMARKS_CSLR)):
                ruta = os.path.join(cfg.DIR_LANDMARKS_CSLR, clase)
                if not os.path.isdir(ruta):
                    continue
                npys = [f for f in os.listdir(ruta) if f.endswith('.npy')]
                if not npys:
                    continue
                
                muestra = np.load(os.path.join(ruta, npys[0]))  # (T, 275)
                
                # Separar en flujos
                pose = muestra[:, :cfg.DIM_FLUJO_POSE]
                manos = muestra[:, cfg.DIM_FLUJO_POSE:]
                cinetico = np.zeros_like(muestra)
                cinetico[1:] = muestra[1:] - muestra[:-1]
                
                if len(entradas) == 3:
                    # Mapear por orden o por nombre
                    tensores = {}
                    for det in entradas:
                        nombre = det['name'].lower()
                        expected_shape = det['shape']
                        
                        if expected_shape[-1] == cfg.DIM_FLUJO_POSE:
                            data = pose
                        elif expected_shape[-1] == cfg.DIM_FLUJO_MANOS:
                            data = manos
                        elif expected_shape[-1] == cfg.DIM_FLUJO_CINETICO:
                            data = cinetico
                        else:
                            # Fallback: match by name
                            if 'pose' in nombre:
                                data = pose
                            elif 'mano' in nombre:
                                data = manos
                            else:
                                data = cinetico
                        
                        interprete.set_tensor(det['index'],
                            np.expand_dims(data, 0).astype(np.float32))
                elif len(entradas) == 1:
                    concatenada = np.concatenate([pose, manos, cinetico], axis=-1)
                    interprete.set_tensor(entradas[0]['index'],
                        np.expand_dims(concatenada, 0).astype(np.float32))
                
                interprete.invoke()
                resultado = interprete.get_tensor(salidas[0]['index'])
                
                idx_pred = np.argmax(resultado[0])
                confianza = resultado[0][idx_pred]
                clase_pred = labels[idx_pred] if idx_pred < len(labels) else "???"
                
                print(f"  Clase real: '{clase}' → Predicción: '{clase_pred}' "
                      f"(confianza: {confianza:.2%})")
                print(f"    Probabilidades: {[f'{p:.2%}' for p in resultado[0]]}")
                break  # Solo 1 muestra por clase para no saturar
        
        # 7. Resumen
        print("\n--- 7. RESUMEN DIAGNÓSTICO ---")
        print(f"  Config LONGITUD_SECUENCIA_CSLR = {cfg.LONGITUD_SECUENCIA_CSLR}")
        print(f"  Config DIM_FEATURE_FRAME = {cfg.DIM_FEATURE_FRAME}")
        print(f"  Config PASO_DESLIZAMIENTO = {cfg.PASO_DESLIZAMIENTO}")
        
        # Verificar si las formas coinciden
        for det in entradas:
            if det['shape'][1] != cfg.LONGITUD_SECUENCIA_CSLR:
                print(f"\n  ⚠ PROBLEMA: La entrada '{det['name']}' espera "
                      f"{det['shape'][1]} frames, pero LONGITUD_SECUENCIA_CSLR = "
                      f"{cfg.LONGITUD_SECUENCIA_CSLR}")
    
    else:
        print("  ✗ No hay modelo TFLite!")


if __name__ == "__main__":
    diagnostico()
