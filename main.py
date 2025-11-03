import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Parámetros
FS = 44100
CHUNK = 8192
A4_FREQ = 440.0
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Suavizado
smoothed = deque(maxlen=5)
def smooth_freq(freq):
    smoothed.append(freq)
    return np.mean(smoothed)

def freq_to_note(freq):
    if freq <= 0:
        return None
    n = 12 * np.log2(freq / A4_FREQ)
    note_index = int(round(n)) + 9
    octave = 4 + note_index // 12
    note_name = NOTES[note_index % 12]
    return f"{note_name}{octave}"

def get_freq_autocorr(data):
    data -= np.mean(data)
    corr = np.correlate(data, data, mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    start = np.where(d > 0)[0]
    if len(start) == 0:
        return 0
    start = start[0]
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return 0
    return FS / peak

# Tkinter UI
root = tk.Tk()
root.title("Afinador en tiempo real con gráfica FFT")

devices = sd.query_devices()
input_devices = [d for d in devices if d['max_input_channels'] > 0]
tk.Label(root, text="Selecciona el dispositivo de entrada:").pack()
device_var = tk.StringVar()
combo = ttk.Combobox(root, textvariable=device_var, state="readonly", width=60)
combo['values'] = [f"{i}: {d['name']}" for i, d in enumerate(input_devices)]
combo.current(0)
combo.pack(pady=5)

label = tk.Label(root, text="—", font=("Arial", 60))
label.pack(padx=20, pady=20)

running = False
current_freq = 0
freq_axis = np.fft.rfftfreq(CHUNK, 1/FS)
fft_data = np.zeros(len(freq_axis))

# Ventana matplotlib
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(freq_axis, np.zeros_like(freq_axis))
ax.set_xlim(0, 1000)
ax.set_ylim(0, 0.1)
ax.set_xlabel("Frecuencia [Hz]")
ax.set_ylabel("Magnitud")
ax.set_title("Transformada de Fourier en tiempo real")

def update_plot(frame):
    line.set_ydata(fft_data)
    return line,

ani = FuncAnimation(fig, update_plot, interval=100, blit=True, cache_frame_data=False)

def start_tuner():
    global running
    if running:
        return
    running = True
    selected = int(device_var.get().split(":")[0])

    def update_note():
        global current_freq, fft_data
        if not running:
            return
        audio = sd.rec(CHUNK, samplerate=FS, channels=1, dtype='float32', device=selected)
        sd.wait()
        data = audio.flatten()

        # FFT para gráfico
        window = np.hanning(len(data))
        fft = np.fft.rfft(data * window)
        fft_data = np.abs(fft) / len(data)

        # Detección de frecuencia por autocorrelación + suavizado
        freq = smooth_freq(get_freq_autocorr(data))
        note = freq_to_note(freq)
        if note:
            label.config(text=f"{note}\n{freq:.1f} Hz")
        root.after(100, update_note)

    update_note()

def stop_tuner():
    global running
    running = False

frame = tk.Frame(root)
frame.pack(pady=10)
tk.Button(frame, text="Iniciar", command=start_tuner).grid(row=0, column=0, padx=5)
tk.Button(frame, text="Detener", command=stop_tuner).grid(row=0, column=1, padx=5)

root.mainloop()
