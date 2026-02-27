import cv2
import os
import time
from . import config

def recolectar_datos_webcam():
    """
    Herramienta para recolectar datos usando la webcam.
    Guarda frames automáticamente en la carpeta del dataset.
    """
    
    print("=== Herramienta de Recolección de Datos ===")
    nombre_clase = input("Ingresa el nombre del gesto/clase (ej. A, B, SALUDO): ").strip().upper()
    
    if not nombre_clase:
        print("Nombre inválido.")
        return

    # Crear directorio si no existe
    ruta_clase = os.path.join(config.DIR_DATOS, nombre_clase)
    if not os.path.exists(ruta_clase):
        os.makedirs(ruta_clase)
        print(f"Carpeta creada: {ruta_clase}")
    else:
        print(f"Usando carpeta existente: {ruta_clase}")

    # Verificar cuántas imágenes ya existen para no sobrescribir
    imgs_existentes = len(os.listdir(ruta_clase))
    contador = imgs_existentes
    print(f"Imágenes existentes: {imgs_existentes}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("\nInstrucciones:")
    print("  [ESPACIO] - Mantén presionado para GUARDAR frames (rápido)")
    print("  [s]       - Presiona una vez para guardar UN solo frame")
    print("  [q]       - Salir")
    print("\nPrepárate... abriendo cámara.")
    
    time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mostrar contadores en pantalla
        cv2.putText(frame, f"Clase: {nombre_clase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Guardadas: {contador}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Dibujar recuadro guía (opcional, para centrar mano)
        h, w, c = frame.shape
        start_x = int(w/2 - 112)
        start_y = int(h/2 - 112)
        cv2.rectangle(frame, (start_x, start_y), (start_x + 224, start_y + 224), (255, 255, 0), 2)

        cv2.imshow('Recoleccion de Datos', frame)
        
        k = cv2.waitKey(1) & 0xFF
        
        guardar = False
        if k == ord('q'):
            break
        elif k == 32: # Espacio mantenido
            guardar = True
        elif k == ord('s'): # 's' presionado
            guardar = True
            
        if guardar:
            # Guardamos la imagen
            # Opcional: Guardar solo el recorte o la imagen completa
            # Para mayor robustez, guardamos la imagen completa y dejamos que el data_loader haga el crop/resize,
            # o hacemos el crop aquí para asegurar calidad. Haremos el crop aquí para coincidir con la inferencia.
            
            # Recorte Centro (mismo que training/inference)
            # Nota: Si prefieres guardar toda la imagen para tener contexto, comenta estas lineas
            # y guarda 'frame'.
            crop_img = frame[start_y:start_y+224, start_x:start_x+224]
            if crop_img.shape[0] != 224 or crop_img.shape[1] != 224:
               # Fallback si el recorte sale de bordes
               crop_img = cv2.resize(frame, (224, 224))

            nombre_archivo = os.path.join(ruta_clase, f"{nombre_clase}_{contador}.jpg")
            
            # Guardar
            cv2.imwrite(nombre_archivo, crop_img)
            contador += 1
            print(f"Guardado: {nombre_archivo}", end='\r')
            time.sleep(0.05) # Pequeña pausa para no saturar si se mantiene espacio

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTerminado. Total para '{nombre_clase}': {contador}")

if __name__ == "__main__":
    recolectar_datos_webcam()
