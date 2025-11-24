import numpy as np           # Librería para cálculos numéricos y manejo eficiente de arreglos/matrices.
import sounddevice as sd     # Permite grabar y reproducir audio desde Python usando dispositivos de sonido.
from collections import deque # Estructura de datos tipo cola doblemente terminada, útil para almacenar historial.
from math import log2         # Función matemática para logaritmo base 2.
import serial                 # Comunicación serie (puertos COM) para interactuar con hardware externo (ej. ESP32).
import serial.tools.list_ports # Herramientas para listar los puertos serie disponibles.
import time                   # Funciones para manejo de tiempo y temporizadores.
import threading              # Permite ejecutar tareas en paralelo (hilos) dentro del programa.

# ---------- PARAMETROS ---------- (ajusta según necesidad)
FS = 44100
CHUNK = 4096
UPDATE_MS = 120
SMOOTH_N = 5
A4_FREQ = 440.0

ORANGE_CENTS = 20
GREEN_CENTS = 5
STABLE_MS_REQUIRED = 500         # ahora 500 ms (0.5 s) de estabilidad requerida
STABLE_CENTS_THRESHOLD = 3.0     # tolerancia en cents para considerar "misma frecuencia" (ajustable)

SOLFEGE = ['Do', 'Do#', 'Re', 'Re#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']

GUITAR_STRINGS = {
    "6 - Mi (E2)": 82.4069,
    "5 - La (A2)": 110.0,
    "4 - Re (D3)": 146.832,
    "3 - Sol (G3)": 195.998,
    "2 - Si (B3)": 246.942,
    "1 - Mi (E4)": 329.628
}

# ---------- SERIAL helpers ----------
def find_esp32_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").upper()
        if "USB" in desc or "ESP32" in desc or "CP210" in desc or "CH340" in desc:
            return p.device
    return None

def open_serial(port, baud=115200, timeout=1):
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)
        ser.flushInput()
        return ser
    except Exception as e:
        print("Error abriendo puerto serial:", e)
        return None

# ---------- FRECUENCIA / NOTA ----------
def freq_to_note_name(freq):
    if freq <= 0 or not np.isfinite(freq):
        return None, None, None
    n = 12 * log2(freq / A4_FREQ)
    semitone = int(round(n))
    note_index = (semitone + 9) % 12
    octave = 4 + ((semitone + 9) // 12)
    note_name = SOLFEGE[note_index]
    note_freq = A4_FREQ * (2 ** (semitone / 12))
    return note_name, octave, note_freq

def cents_difference(freq, target_freq):
    if freq <= 0 or target_freq <= 0:
        return None
    return 1200 * log2(freq / target_freq)

def get_freq_autocorr(data):
    data = np.nan_to_num(np.asarray(data, dtype=float))
    if np.allclose(data, 0):
        return 0.0
    data -= np.mean(data)
    window = np.hanning(len(data))
    data_w = data * window
    corr = np.correlate(data_w, data_w, mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    try:
        start = np.where(d > 0)[0][0]
    except IndexError:
        return 0.0
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return 0.0
    return FS / peak

# ---------- MOTOR CONTROLLER (protocolo simple) ----------
class MotorController:
    """
    Protocolo: send "<dir><steps>\n" where dir is '+' (tensionar) or '-' (aflojar).
    ESP32 replies "DONE\n" when finished. Send "S\n" to stop/abort.
    """
    def __init__(self, ser):
        self.ser = ser
        self.lock = threading.Lock()
        self.last_response = None
        self._running = False
        if ser:
            self._running = True
            t = threading.Thread(target=self._reader_thread, daemon=True)
            t.start()

    def _reader_thread(self):
        while self._running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    with self.lock:
                        self.last_response = line
            except Exception:
                pass
            time.sleep(0.01)

    def send_move(self, direction, steps, timeout=10.0):
        if not self.ser or not self.ser.is_open:
            return False
        cmd = f"{direction}{int(steps)}\n"
        with self.lock:
            self.last_response = None
        try:
            self.ser.write(cmd.encode('utf-8'))
        except Exception:
            return False
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.last_response is not None:
                    if "DONE" in self.last_response:
                        return True
            time.sleep(0.02)
        return False

    def stop(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b"S\n")
            except Exception:
                pass

    def close(self):
        self._running = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


# Lanzar la interfaz gráfica desde el archivo interfaz.py
if __name__ == "__main__":
    from interfaz import TunerApp
    import tkinter as tk
    root = tk.Tk()
    app = TunerApp(root)
    root.mainloop()
