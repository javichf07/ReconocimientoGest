from collections import deque, Counter

class ProcesadorTiempoReal:
    def __init__(self, tamano_buffer=10, umbral_estabilidad=8, frames_enfriamiento=20):
        self.tamano_buffer = tamano_buffer # Frames a mantener para suavizado
        self.buffer_prediccion = deque(maxlen=tamano_buffer)
        self.umbral_estabilidad = umbral_estabilidad # Cuántos frames iguales en el buffer para confirmar
        self.frames_enfriamiento = frames_enfriamiento # Espera antes de añadir la misma palabra de nuevo
        self.enfriamiento_actual = 0
        
        self.oracion = []
        self.ultima_palabra = ""
        
    def procesar_prediccion(self, etiqueta_prediccion):
        """
        Toma una nueva predicción (etiqueta string) y actualiza el estado.
        Retorna la oración actual como string.
        """
        self.buffer_prediccion.append(etiqueta_prediccion)
        
        # Manejar Cooldown (Enfriamiento)
        if self.enfriamiento_actual > 0:
            self.enfriamiento_actual -= 1
            return " ".join(self.oracion)
            
        # Revisar estabilidad
        if len(self.buffer_prediccion) == self.tamano_buffer:
            # Contar ocurrencias en buffer
            conteos = Counter(self.buffer_prediccion)
            mas_comun, cantidad = conteos.most_common(1)[0]
            
            if cantidad >= self.umbral_estabilidad:
                # Gesto confirmado
                # Asumimos que "background" y "nothing" son clases para fondo/nada
                if mas_comun != "background" and mas_comun != "nothing": 
                    # Lógica para evitar repetición rápida
                    if mas_comun != self.ultima_palabra:
                        self._agregar_palabra(mas_comun)
                        self.enfriamiento_actual = self.frames_enfriamiento
                        self.ultima_palabra = mas_comun
                    
        return " ".join(self.oracion)
    
    def _agregar_palabra(self, palabra):
        if palabra == "SPACE":
            # Si la clase es SPACE, agregamos un espacio real si es necesario o no hacemos nada si solo unimos
            self.oracion.append(" ") 
            pass
        elif palabra == "DELETE":
            if self.oracion:
                self.oracion.pop()
        else:
            self.oracion.append(palabra)

    def obtener_oracion(self):
        return " ".join(self.oracion)
    
    def limpiar(self):
        self.oracion = []
        self.ultima_palabra = ""
        self.buffer_prediccion.clear()
