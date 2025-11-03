# tuner_with_guitar_mode.py
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
from math import log2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------- PARAMETROS ----------
FS = 44100
CHUNK = 4096            # mayor CHUNK => mejor resolución baja frecuencia
UPDATE_MS = 120         # intervalo UI en ms
SMOOTH_N = 5            # número de lecturas a promediar
A4_FREQ = 440.0

# Umbrales en cents
ORANGE_CENTS = 20       # dentro de +/- => naranja (aproximando)
GREEN_CENTS = 5         # dentro de +/- => condición de afinado
STABLE_MS_REQUIRED = 1000  # ms sostenidos para ponerse verde

# Notas en solfeo (Do = C)
SOLFEGE = ['Do', 'Do#', 'Re', 'Re#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']

# Afinación estándar de guitarra (cuerdas abiertas, de grave a agudo)
GUITAR_STRINGS = {
    "6 - Mi (E2)": 82.4068892282175,
    "5 - La (A2)": 110.000000000000,
    "4 - Re (D3)": 146.8323839587038,
    "3 - Sol (G3)": 195.9977179908746,
    "2 - Si (B3)": 246.9416506280621,
    "1 - Mi (E4)": 329.6275569128699
}

# ---------- UTILIDADES ----------
def freq_to_note_name(freq):
    """Convierte frecuencia a nombre de nota y octava (solfeo)."""
    if freq <= 0 or not np.isfinite(freq):
        return None, None, None
    # número de semitonos relativo a A4
    n = 12 * log2(freq / A4_FREQ)
    semitone = int(round(n))
    note_index = (semitone + 9) % 12  # A -> índice 9
    octave = 4 + ((semitone + 9) // 12)
    note_name = SOLFEGE[note_index]
    # frecuencia exacta de la nota redondeada
    note_freq = A4_FREQ * (2 ** (semitone / 12))
    return note_name, octave, note_freq

def cents_difference(freq, target_freq):
    if freq <= 0 or target_freq <= 0:
        return None
    return 1200 * log2(freq / target_freq)

def get_freq_autocorr(data):
    """Estimación de la frecuencia fundamental por autocorrelación robusta."""
    # eliminar DC y vigilar valores nulos
    data = np.asarray(data, dtype=float)
    if np.allclose(data, 0):
        return 0.0
    data -= np.mean(data)
    # ventana para reducir bordes
    window = np.hanning(len(data))
    data_w = data * window
    corr = np.correlate(data_w, data_w, mode='full')
    corr = corr[len(corr)//2:]  # mitad positiva
    # buscar primer cruce positivo para ignorar pico en cero
    d = np.diff(corr)
    try:
        start = np.where(d > 0)[0][0]
    except IndexError:
        return 0.0
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return 0.0
    return FS / peak

# ---------- INTERFAZ ----------
class TunerApp:
    def __init__(self, root):
        self.root = root
        root.title("Afinador - FFT + Afinador de Guitarra")
        self.running = False
        self.device_index = None

        # smoothing
        self.history = deque(maxlen=SMOOTH_N)

        # guitarra mode
        self.mode_var = tk.StringVar(value="Normal")
        self.string_var = tk.StringVar()
        self.completed_strings = set()
        self.stable_count = 0
        self.completed_label_var = tk.StringVar(value="No completadas")

        # FFT data
        self.freq_axis = np.fft.rfftfreq(CHUNK, 1/FS)
        self.fft_data = np.zeros(len(self.freq_axis))

        self.build_ui()
        self.populate_devices()

    def build_ui(self):
        frm = ttk.Frame(self.root)
        frm.pack(padx=8, pady=8, fill='x')

        # fila: dispositivo + modo
        top = ttk.Frame(frm)
        top.pack(fill='x', pady=4)
        ttk.Label(top, text="Dispositivo entrada:").grid(row=0, column=0, sticky='w')
        self.device_combo = ttk.Combobox(top, state='readonly', width=60)
        self.device_combo.grid(row=0, column=1, padx=6)
        ttk.Label(top, text="Modo:").grid(row=0, column=2, sticky='e', padx=(10,0))
        mode_combo = ttk.Combobox(top, state='readonly', values=["Normal", "Afinador guitarra"], textvariable=self.mode_var, width=20)
        mode_combo.grid(row=0, column=3, padx=6)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

        # fila: selector cuerda (solo guitarra)
        string_row = ttk.Frame(frm)
        string_row.pack(fill='x', pady=2)
        ttk.Label(string_row, text="Cuerda (guitarra):").grid(row=0, column=0, sticky='w')
        self.string_combo = ttk.Combobox(string_row, state='readonly', textvariable=self.string_var, width=30)
        self.string_combo['values'] = list(GUITAR_STRINGS.keys())
        self.string_combo.grid(row=0, column=1, padx=6)
        self.string_combo.current(0)
        self.string_combo.config(state='disabled')

        # fila: botones
        btns = ttk.Frame(frm)
        btns.pack(fill='x', pady=6)
        self.start_btn = ttk.Button(btns, text="Iniciar", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Detener", command=self.stop).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Reiniciar completadas", command=self.reset_completed).grid(row=0, column=2, padx=6)

        # fila: label nota grande
        self.note_label = tk.Label(self.root, text="—", font=("Arial", 44), width=16)
        self.note_label.pack(pady=8)

        # fila: detalles
        details = ttk.Frame(self.root)
        details.pack(fill='x')
        self.freq_var = tk.StringVar(value="Freq: - Hz")
        self.cents_var = tk.StringVar(value="Cents: -")
        ttk.Label(details, textvariable=self.freq_var).grid(row=0, column=0, padx=6)
        ttk.Label(details, textvariable=self.cents_var).grid(row=0, column=1, padx=6)
        ttk.Label(details, text="Completadas:").grid(row=0, column=2, padx=(20,2))
        ttk.Label(details, textvariable=self.completed_label_var).grid(row=0, column=3, padx=2)

        # area grafica (matplotlib embebido)
        fig = Figure(figsize=(7,3))
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Frecuencia [Hz]")
        self.ax.set_ylabel("Magnitud")
        self.ax.set_title("FFT (rango 60-2000 Hz)")
        self.line, = self.ax.plot(self.freq_axis, self.fft_data)
        self.ax.set_xlim(60, 2000)
        self.ax.set_ylim(0, 1e-6)
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        fig.tight_layout()

    def populate_devices(self):
        devs = sd.query_devices()
        in_devs = [(i, d['name']) for i, d in enumerate(devs) if d['max_input_channels'] > 0]
        if not in_devs:
            messagebox.showerror("Error", "No se encontró dispositivo de entrada con canales.")
            return
        values = [f"{i}: {name}" for i, name in in_devs]
        self.device_map = {f"{i}: {name}": i for i, name in in_devs}
        self.device_combo['values'] = values
        self.device_combo.current(0)
        # actualizar device_index
        self.device_index = self.device_map[self.device_combo.get()]

    def on_mode_change(self, _ev=None):
        if self.mode_var.get() == "Afinador guitarra":
            self.string_combo.config(state='readonly')
        else:
            self.string_combo.config(state='disabled')

    def start(self):
        if self.running:
            return
        try:
            sel = self.device_combo.get()
            self.device_index = self.device_map[sel]
        except Exception:
            messagebox.showerror("Error", "Selecciona un dispositivo válido")
            return
        self.running = True
        self.root.after(10, self.update_loop)

    def stop(self):
        self.running = False

    def reset_completed(self):
        self.completed_strings.clear()
        self.update_completed_label()

    def update_completed_label(self):
        remaining = [k for k in GUITAR_STRINGS.keys() if k not in self.completed_strings]
        self.completed_label_var.set(", ".join(remaining) if remaining else "Todas completadas")

    def update_loop(self):
        if not self.running:
            return
        try:
            audio = sd.rec(CHUNK, samplerate=FS, channels=1, dtype='float32', device=self.device_index)
            sd.wait()
            data = audio.flatten()
        except Exception as e:
            self.note_label.config(text="Error", fg="red")
            self.freq_var.set(f"Error: {e}")
            self.root.after(UPDATE_MS, self.update_loop)
            return

        # FFT para gráfica
        window = np.hanning(len(data))
        fft = np.fft.rfft(data * window)
        mag = np.abs(fft) / len(data)
        self.fft_data = mag
        # actualizar gráfica (limitar y escalar)
        self.line.set_ydata(self.fft_data)
        self.ax.set_ylim(0, max(1e-6, self.fft_data.max()*1.2))
        self.canvas.draw_idle()

        # estimación de frecuencia con autocorrelación + suavizado
        freq = get_freq_autocorr(data)
        if freq <= 0:
            # si no detecta, mostrar guion
            self.note_label.config(text="—", fg="black")
            self.freq_var.set("Freq: - Hz")
            self.cents_var.set("Cents: -")
            self.root.after(UPDATE_MS, self.update_loop)
            return

        # suavizado por media móvil
        self.history.append(freq)
        freq_s = float(np.mean(self.history))

        # modo guitarra: objetivo específico
        if self.mode_var.get() == "Afinador guitarra":
            sel_string = self.string_var.get()
            if sel_string in self.completed_strings:
                # marcar completada y saltar
                self.note_label.config(text=f"{sel_string}\nCompletada", fg="gray")
                self.update_completed_label()
                self.freq_var.set(f"Freq: {freq_s:.1f} Hz")
                self.cents_var.set("Cents: 0")
                self.root.after(UPDATE_MS, self.update_loop)
                return

            target_freq = GUITAR_STRINGS.get(sel_string)
            cents = cents_difference(freq_s, target_freq)
            self.freq_var.set(f"Freq: {freq_s:.1f} Hz (obj: {target_freq:.1f} Hz)")
            self.cents_var.set(f"Cents: {cents:+.1f}")

            # determino acción tensar/aflojar
            if cents is None:
                action = ""
            elif cents < -ORANGE_CENTS:
                action = "Grave → tensar cuerda"
            elif cents > ORANGE_CENTS:
                action = "Agudo → aflojar cuerda"
            else:
                action = "Ajustando..."

            # color lógico y condición de completado
            if abs(cents) <= GREEN_CENTS:
                self.stable_count += 1
            else:
                self.stable_count = 0

            # naranja si dentro de ORANGE_CENTS pero fuera de GREEN_CENTS
            if abs(cents) <= ORANGE_CENTS and abs(cents) > GREEN_CENTS:
                color = "orange"
            elif abs(cents) <= GREEN_CENTS and (self.stable_count * UPDATE_MS) >= STABLE_MS_REQUIRED:
                color = "green"
                # marcar completado
                self.completed_strings.add(sel_string)
                self.update_completed_label()
            else:
                color = "red"

            # mostrar nota detectada y acción
            note_name, octave, note_freq = freq_to_note_name(freq_s)
            display = f"{note_name}{octave}\n{action}"
            self.note_label.config(text=display, fg=color)
            # actualizar textos
            self.root.after(UPDATE_MS, self.update_loop)
            return

        # modo normal: mostrar nota detectada (solfeo)
        note_name, octave, note_freq = freq_to_note_name(freq_s)
        cents_to_note = cents_difference(freq_s, note_freq) if note_freq else None
        self.freq_var.set(f"Freq: {freq_s:.1f} Hz")
        if cents_to_note is None:
            self.cents_var.set("Cents: -")
        else:
            self.cents_var.set(f"Cents to {note_name}{octave}: {cents_to_note:+.1f}")
        # color por cercanía al semitono más cercano
        color = "green" if (cents_to_note is not None and abs(cents_to_note) <= GREEN_CENTS) else ("orange" if (cents_to_note is not None and abs(cents_to_note) <= ORANGE_CENTS) else "black")
        self.note_label.config(text=f"{note_name}{octave}", fg=color)

        self.root.after(UPDATE_MS, self.update_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = TunerApp(root)
    root.mainloop()
