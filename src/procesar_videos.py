import cv2
import os
import glob
from . import config

def extraer_frames_de_videos(origen_videos="videos", destino_dataset=config.DIR_DATOS, frames_por_segundo=5):
    """
    Recorre la carpeta de videos y extrae frames para crear el dataset de imágenes.
    
    Estructura esperada en 'origen_videos':
        videos/
          A/
            video1.mp4
            video2.mov
          B/
            ...
            
    Salida en 'dataset':
        dataset/
          A/
            video1_frame0.jpg
            ...
    """
    
    ruta_base_videos = os.path.join(os.path.dirname(os.path.dirname(__file__)), origen_videos)
    
    if not os.path.exists(ruta_base_videos):
        print(f"No se encontró la carpeta de videos en: {ruta_base_videos}")
        print("Crea carpetas por cada clase (A, B, C...) dentro de 'videos' y pon tus archivos allí.")
        return

    print(f"Buscando videos en: {ruta_base_videos}")
    
    # Recorrer cada subcarpeta (Clase)
    clases = [d for d in os.listdir(ruta_base_videos) if os.path.isdir(os.path.join(ruta_base_videos, d))]
    
    if not clases:
        print("No se encontraron carpetas de clases. Asegúrate de organizar los videos en carpetas: videos/A/, videos/B/...")
        return

    imgs_totales = 0
    
    for clase in clases:
        ruta_clase_videos = os.path.join(ruta_base_videos, clase)
        ruta_clase_destino = os.path.join(destino_dataset, clase)
        
        if not os.path.exists(ruta_clase_destino):
            os.makedirs(ruta_clase_destino)
            
        # Buscar archivos de video (mp4, avi, mov, mkv)
        archivos_video = []
        extensiones = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        for ext in extensiones:
            archivos_video.extend(glob.glob(os.path.join(ruta_clase_videos, ext)))
            
        print(f"Procesando clase '{clase}': {len(archivos_video)} videos encontrados.")
        
        for ruta_video in archivos_video:
            nombre_video = os.path.splitext(os.path.basename(ruta_video))[0]
            
            # Intento normal de abrir video
            cap = cv2.VideoCapture(ruta_video)
            
            # Si falla (común en Windows con tildes/ñ), intentar workaround con archivo temporal
            usando_temp = False
            ruta_temp = "temp_video_workaround.mp4"
            
            if not cap.isOpened():
                print(f"Advertencia: No se pudo abrir '{nombre_video}' directamente (posible caracter especial).")
                print("Intentando workaround con archivo temporal...")
                
                try:
                    import shutil
                    shutil.copy2(ruta_video, ruta_temp)
                    cap = cv2.VideoCapture(ruta_temp)
                    usando_temp = True
                    if not cap.isOpened():
                        print(f"Error: Aún no se puede abrir el video {nombre_video}.")
                        if os.path.exists(ruta_temp): os.remove(ruta_temp)
                        continue
                except Exception as e:
                    print(f"Error creando archivo temporal: {e}")
                    continue
            
            
            fps_original = cap.get(cv2.CAP_PROP_FPS)
            if fps_original == 0: fps_original = 30 # Fallback
            
            # Calcular cada cuántos frames capturar para cumplir con frames_por_segundo deseados
            salto_frames = int(fps_original / frames_por_segundo)
            if salto_frames < 1: salto_frames = 1
            
            contador_frame = 0
            guardados = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if contador_frame % salto_frames == 0:
                    # Guardar frame
                    nombre_img = f"{nombre_video}_{contador_frame}.jpg"
                    ruta_img = os.path.join(ruta_clase_destino, nombre_img)
                    
                    # Opcional: Redimensionar aquí si los videos son 4K para ahorrar espacio, 
                    # pero data_loader ya lo hace al entrenar.
                    # frame = cv2.resize(frame, (config.ANCHO_IMG, config.ALTO_IMG))
                    
                    cv2.imwrite(ruta_img, frame)
                    guardados += 1
                    imgs_totales += 1
                
                contador_frame += 1
                
            cap.release()
            
            if usando_temp and os.path.exists(ruta_temp):
                try:
                    os.remove(ruta_temp)
                except:
                    pass
                    
            print(f"  -> Extrados {guardados} imágenes de {nombre_video}")

    print(f"\nProceso completado. Total imágenes generadas: {imgs_totales}")

if __name__ == "__main__":
    # Se puedeajustar frames_por_segundo segunse requiera
    extraer_frames_de_videos(frames_por_segundo=30)
