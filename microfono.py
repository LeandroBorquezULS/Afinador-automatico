import threading
import numpy as np
import sounddevice as sd

def probar_nivel_microfono(indice_microfono, barra_nivel, nivel_var):
    nivel_stop = threading.Event()
    def _nivel():
        try:
            with sd.InputStream(device=indice_microfono, channels=1, samplerate=44100) as stream:
                while not nivel_stop.is_set():
                    audio, _ = stream.read(1024)
                    if nivel_stop.is_set():
                        break
                    nivel = np.abs(audio).mean()
                    valor = int(nivel * 5000)
                    barra_nivel['value'] = valor
                    nivel_var.set(f"Nivel: {valor}")
        except Exception as e:
            nivel_var.set(f"Error: {e}")
    nivel_thread = threading.Thread(target=_nivel, daemon=True)
    nivel_thread.start()
    return nivel_thread, nivel_stop
