import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sounddevice as sd
from collections import deque
import threading
import time
from PIL import Image, ImageTk
import os
from microfono import probar_nivel_microfono

# Importa todos los parámetros globales necesarios desde main.py
from main import (
    SMOOTH_N, FS, CHUNK, UPDATE_MS, GREEN_CENTS, ORANGE_CENTS, STABLE_MS_REQUIRED,
    STABLE_CENTS_THRESHOLD, A4_FREQ, GUITAR_STRINGS, freq_to_note_name, cents_difference,
    get_freq_autocorr, find_esp32_port, open_serial, MotorController
)

class TunerApp:
    def __init__(self, root):
        self.root = root
        root.title("Afinador con control de ESP32")
        # --- ICONO DE LA VENTANA PRINCIPAL ---
        try:
            # Usa un archivo .ico para Windows
            root.iconbitmap("icono.ico")
        except Exception:
            # Alternativa multiplataforma (requiere icono en PNG)
            try:
                icon_img = tk.PhotoImage(file="icono.png")
                root.iconphoto(True, icon_img)
            except Exception:
                pass  # Si no hay icono, sigue sin error

        self.running = False
        self.device_index = None
        self.device_map = {}  # Asegura que device_map esté disponible
        self.history = deque(maxlen=SMOOTH_N)
        self.mode_var = tk.StringVar(value="Normal")
        self.string_var = tk.StringVar()
        self.completed_strings = set()
        self.cents_per_step_var = tk.DoubleVar(value=1.0)  # ajustar mediante calibración
        self.max_steps_var = tk.IntVar(value=50)          # "n" inicial por acción
        self.step_timeout_var = tk.DoubleVar(value=8.0)
        self.motor_enabled_var = tk.BooleanVar(value=True)

        self.freq_axis = np.fft.rfftfreq(CHUNK, 1/FS)
        self.fft_data = np.zeros(len(self.freq_axis))

        self.motor = None
        self.ser = None

        # estabilidad
        self._stable_candidate_freq = None
        self._stable_since = None

        # --- CARGA DE ICONOS PARA BOTONES ---
        self.icon_play = None
        self.icon_stop = None
        self.icon_reset = None
        self.icon_tocar = None
        self.icon_boton_detener = None
        self.icon_logo = None  # Para el PNG decorativo
        try:
            self.icon_play = tk.PhotoImage(file="play.png")
        except Exception:
            pass
        try:
            self.icon_stop = tk.PhotoImage(file="stop.png")
        except Exception:
            pass
        try:
            self.icon_reset = tk.PhotoImage(file="reset.png")
        except Exception:
            pass
        try:
            self.icon_tocar = tk.PhotoImage(file="tocar.png")
        except Exception:
            pass
        try:
            self.icon_boton_detener = tk.PhotoImage(file="boton-detener.png")
        except Exception:
            pass
        try:
            self.icon_logo = tk.PhotoImage(file="logo.png")  # Cambia "logo.png" por el nombre de tu archivo
        except Exception:
            self.icon_logo = None

        iconos_path = os.path.join(os.path.dirname(__file__), "Iconos")
        self.img_tocar = ImageTk.PhotoImage(Image.open(os.path.join(iconos_path, "tocar.png")).resize((48, 48)))
        self.img_detener = ImageTk.PhotoImage(Image.open(os.path.join(iconos_path, "boton-detener.png")).resize((48, 48)))
        self.img_ajuste = ImageTk.PhotoImage(Image.open(os.path.join(iconos_path, "ajuste.png")).resize((32, 32)))
        self.img_microfono = ImageTk.PhotoImage(Image.open(os.path.join(iconos_path, "microfono.png")).resize((24, 24)))
        # Corrige la extensión del icono de reiniciar a .png
        self.img_reiniciar = ImageTk.PhotoImage(
            Image.open(os.path.join(iconos_path, "rotacion-de-flecha-circular-en-sentido-antihorario.png")).resize((32, 32))
        )

        self.build_ui()
        self.populate_devices()
        self.try_open_serial()
        self.advanced_vars = {}

        # --- BOTÓN TOGGLE INICIAR/DETENER (icono PNG) ---
        # self.boton_toggle = tk.Button(root, image=self.img_tocar, command=self.toggle_iniciar_detener)
        # self.boton_toggle.image = self.img_tocar
        # self.boton_toggle.pack()

        # Elimina los botones de tkinter de opciones avanzadas y de iniciar/detener (solo deja los iconos PNG):
        # self.start_btn = ttk.Button(...)  # ELIMINADO
        # self.boton_config = tk.Button(...)  # ELIMINADO si era de texto, deja solo el de imagen si lo tienes
        # adv_btn = ttk.Button(...)  # ELIMINADO

        # Estado interno para saber si está corriendo
        self._esta_iniciando = False

        self.nivel_var = tk.StringVar(value="Nivel: 0")
        self.barra_nivel = ttk.Progressbar(self.root, orient="horizontal", length=250, mode="determinate", maximum=1000)
        self.nivel_label = ttk.Label(self.root, textvariable=self.nivel_var)
        self.nivel_thread = None
        self.nivel_stop = None

    def toggle_iniciar_detener(self):
        if not self._esta_iniciando:
            self.iniciar()
            self.boton_toggle.config(image=self.img_detener, command=self.toggle_iniciar_detener)
            self.boton_toggle.image = self.img_detener
            self._esta_iniciando = True
        else:
            self.detener()
            self.boton_toggle.config(image=self.img_tocar, command=self.toggle_iniciar_detener)
            self.boton_toggle.image = self.img_tocar
            self._esta_iniciando = False

    def iniciar(self):
        self.start()
        # Inicia la barra de nivel del micrófono
        try:
            indice_microfono = self.device_index if self.device_index is not None else 0
            self.nivel_thread, self.nivel_stop = probar_nivel_microfono(indice_microfono, self.barra_nivel, self.nivel_var)
        except Exception as e:
            self.nivel_var.set(f"Error: {e}")

    def detener(self):
        self.stop()
        # Detiene y limpia la barra de nivel del micrófono
        try:
            if self.nivel_stop is not None:
                self.nivel_stop.set()
            if self.nivel_thread is not None and self.nivel_thread.is_alive():
                self.nivel_thread.join(timeout=1)
        except Exception:
            pass
        self.nivel_thread = None
        self.nivel_stop = None
        self.barra_nivel['value'] = 0
        self.nivel_var.set("Nivel: 0")
        # ...ya se limpia el gráfico y notas en stop()...

    def configuracion_avanzada(self):
        self.open_advanced_options()

    def build_ui(self):
        frm = ttk.Frame(self.root)
        frm.pack(padx=8, pady=8, fill='x')

        # --- AGREGA EL PNG DECORATIVO EN LA PARTE SUPERIOR ---
        if self.icon_logo:
            logo_label = tk.Label(frm, image=self.icon_logo)
            logo_label.pack(pady=(0, 8))

        top = ttk.Frame(frm)
        top.pack(fill='x', pady=4)
        # --- AGREGA EL ICONO DE MICROFONO EN VEZ DEL TEXTO ---
        tk.Label(top, image=self.img_microfono).grid(row=0, column=0, sticky='w', padx=(0, 4))
        self.device_combo = ttk.Combobox(top, state='readonly', width=60)
        self.device_combo.grid(row=0, column=1, padx=6)
        ttk.Label(top, text="Modo:").grid(row=0, column=2, sticky='e', padx=(10,0))
        mode_combo = ttk.Combobox(top, state='readonly', values=["Normal", "Afinador guitarra"], textvariable=self.mode_var, width=20)
        mode_combo.grid(row=0, column=3, padx=6)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

        string_row = ttk.Frame(frm)
        string_row.pack(fill='x', pady=2)
        ttk.Label(string_row, text="Cuerda:").grid(row=0, column=0, sticky='w')
        self.string_combo = ttk.Combobox(string_row, state='readonly', textvariable=self.string_var, width=30)
        self.string_combo['values'] = list(GUITAR_STRINGS.keys())
        self.string_combo.grid(row=0, column=1, padx=6)
        self.string_combo.current(0)
        self.string_combo.config(state='disabled')

        conf_row = ttk.Frame(frm)
        conf_row.pack(fill='x', pady=2)
        # --- QUITA ESTOS CAMPOS DE LA INTERFAZ PRINCIPAL ---
        # ttk.Label(conf_row, text="Cents/step (calibrar):").grid(row=0, column=0, sticky='w')
        # ttk.Entry(conf_row, textvariable=self.cents_per_step_var, width=8).grid(row=0, column=1, padx=4)
        # ttk.Label(conf_row, text="Max steps/action (n):").grid(row=0, column=2, sticky='w', padx=(8,0))
        # ttk.Entry(conf_row, textvariable=self.max_steps_var, width=6).grid(row=0, column=3, padx=4)
        # ttk.Label(conf_row, text="Step timeout (s):").grid(row=0, column=4, sticky='w', padx=(8,0))
        # ttk.Entry(conf_row, textvariable=self.step_timeout_var, width=6).grid(row=0, column=5, padx=4)
        # ttk.Checkbutton(conf_row, text="Motor enabled", variable=self.motor_enabled_var).grid(row=0, column=6, padx=(8,0))

        btns = ttk.Frame(frm)
        btns.pack(fill='x', pady=6)
        # --- Botón único de tocar/detener (cambia de imagen según estado) ---
        self.boton_toggle = tk.Button(
            btns,
            image=self.img_tocar,
            command=self.toggle_iniciar_detener,
            borderwidth=0,
            highlightthickness=0,
            relief="flat"
        )
        self.boton_toggle.grid(row=0, column=0, padx=6)
        # Botón de reiniciar completadas
        tk.Button(
            btns,
            image=self.img_reiniciar,
            command=self.reset_completed,
            borderwidth=0,
            highlightthickness=0,
            relief="flat"
        ).grid(row=0, column=1, padx=6)
        # Botón de ajuste
        tk.Button(
            btns,
            image=self.img_ajuste,
            command=self.configuracion_avanzada,
            borderwidth=0,
            highlightthickness=0,
            relief="flat"
        ).grid(row=0, column=2, padx=6)

        # 1) Widget de notas (self.note_label)
        self.note_label = tk.Label(self.root, text="—", font=("Arial", 44), width=16)
        self.note_label.pack(pady=8)

        # 2) Barra del micrófono (self.barra_nivel y self.nivel_label)
        self.barra_nivel = ttk.Progressbar(self.root, orient="horizontal", length=250, mode="determinate", maximum=1000)
        self.nivel_var = tk.StringVar(value="Nivel: 0")
        self.nivel_label = ttk.Label(self.root, textvariable=self.nivel_var)
        self.barra_nivel.pack(pady=(2, 0))
        self.nivel_label.pack()

        # --- Mueve el frame de detalles aquí ---
        details = ttk.Frame(self.root)
        details.pack(fill='x')
        self.freq_var = tk.StringVar(value="Freq: - Hz")
        self.cents_var = tk.StringVar(value="Cents: -")
        self.completed_label_var = tk.StringVar(value="No completadas")
        ttk.Label(details, textvariable=self.freq_var).grid(row=0, column=0, padx=6)
        ttk.Label(details, textvariable=self.cents_var).grid(row=0, column=1, padx=6)
        ttk.Label(details, text="Completadas:").grid(row=0, column=2, padx=(20,2))
        ttk.Label(details, textvariable=self.completed_label_var).grid(row=0, column=3, padx=2)

        # 3) Gráfica (self.canvas)
        fig = Figure(figsize=(7,3))
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Frecuencia [Hz]")
        self.ax.set_ylabel("Magnitud")
        self.ax.set_title("FFT (60-2000 Hz)")
        self.line, = self.ax.plot(self.freq_axis, self.fft_data)
        self.ax.set_xlim(60, 2000)
        self.ax.set_ylim(0, 1e-6)
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        fig.tight_layout()

        # detalles = ttk.Frame(self.root)
        # detalles.pack(fill='x')
        # self.freq_var = tk.StringVar(value="Freq: - Hz")
        # self.cents_var = tk.StringVar(value="Cents: -")
        # self.completed_label_var = tk.StringVar(value="No completadas")
        # ttk.Label(detalles, textvariable=self.freq_var).grid(row=0, column=0, padx=6)
        # ttk.Label(detalles, textvariable=self.cents_var).grid(row=0, column=1, padx=6)
        # ttk.Label(detalles, text="Completadas:").grid(row=0, column=2, padx=(20,2))
        # ttk.Label(detalles, textvariable=self.completed_label_var).grid(row=0, column=3, padx=2)

    def populate_devices(self):
        devs = sd.query_devices()
        in_devs = []
        for i, d in enumerate(devs):
            if isinstance(d, dict) and d.get('max_input_channels', 0) > 0:
                in_devs.append((i, d.get('name', f"Device {i}")))
        values = [f"{i}: {name}" for i, name in in_devs]
        self.device_map = {f"{i}: {name}": i for i, name in in_devs}
        self.device_combo['values'] = values  # <-- vuelve a agregar esta línea
        if values:
            self.device_combo.current(0)
            self.device_index = self.device_map[self.device_combo.get()]
        # Guarda la lista de dispositivos para opciones avanzadas
        self._input_devices = values

    def try_open_serial(self):
        port = find_esp32_port()
        if port is None:
            print("No se encontró ESP32")
            self.ser = None
            self.motor = None
        else:
            self.ser = open_serial(port, 115200, timeout=0.1)
            if self.ser:
                self.motor = MotorController(self.ser)
                print("Conectado a", port)
            else:
                self.motor = None

    def on_mode_change(self, _ev=None):
        if self.mode_var.get() == "Afinador guitarra":
            self.string_combo.config(state='readonly')
        else:
            self.string_combo.config(state='disabled')

    def toggle_start_stop(self):
        if not self.running:
            # Cambia icono y texto a "tocar.png" y "Detener"
            if self.icon_tocar:
                self.start_btn.config(image=self.icon_tocar, text="Detener")
            else:
                self.start_btn.config(text="Detener")
            self.start()
        else:
            # Cambia icono y texto a "boton detener.png" y "Iniciar"
            if self.icon_boton_detener:
                self.start_btn.config(image=self.icon_boton_detener, text="Iniciar")
            elif self.icon_play:
                self.start_btn.config(image=self.icon_play, text="Iniciar")
            else:
                self.start_btn.config(text="Iniciar")
            self.stop()

    def start(self):
        if self.running:
            return
        try:
            sel = self.device_combo.get()
            self.device_index = self.device_map[sel]
        except Exception:
            messagebox.showerror("Error", "Selecciona un dispositivo válido")
            return
        if self.motor_enabled_var.get() and not self.motor:
            self.try_open_serial()
        self.running = True
        self.root.after(10, self.update_loop)

    def stop(self):
        self.running = False
        if self.motor:
            self.motor.stop()
        # Elimina cualquier referencia a self.start_btn y self.stop_btn, ya que no existen:
        # if self.icon_play:
        #     self.start_btn.config(image=self.icon_play, text="Iniciar")
        # else:
        #     self.start_btn.config(text="Iniciar")
        # if self.icon_stop:
        #     self.stop_btn.config(image=self.icon_stop)
        # --- LIMPIA LA INTERFAZ ---
        self.fft_data = np.zeros(len(self.freq_axis))
        self.line.set_ydata(self.fft_data)
        self.ax.set_ylim(0, 1e-6)
        self.canvas.draw_idle()
        self.note_label.config(text="—", fg="black")
        self.freq_var.set("Freq: - Hz")
        self.cents_var.set("Cents: -")
        self.barra_nivel['value'] = 0
        self.nivel_var.set("Nivel: 0")

    def reset_completed(self):
        self.completed_strings.clear()
        self.update_completed_label()

    def update_completed_label(self):
        remaining = [k for k in GUITAR_STRINGS.keys() if k not in self.completed_strings]
        self.completed_label_var.set(", ".join(remaining) if remaining else "Todas completadas")

    def _move_and_wait(self, direction_sign, steps):
        if not self.motor:
            return False
        timeout = float(self.step_timeout_var.get())
        ok = self.motor.send_move(direction_sign, steps, timeout=timeout)
        return ok

    def iterative_tune(self, initial_cents, initial_sign):
        """
        Algoritmo iterativo con reducción de pasos por overshoot:
         - initial_n = max_steps (parámetro)
         - steps a mover inicialmente = min(initial_n, rounding(initial_cents / cents_per_step)) o initial_n si esto da 0
         - si overshoot -> reverse y reducir pasos según n / (m^2) (m = número de overshoots/iteraciones)
         - detener cuando abs(cents) <= GREEN_CENTS
        """
        if not self.motor or not self.motor_enabled_var.get():
            return

        cents_per_step = float(self.cents_per_step_var.get()) or 1.0
        initial_n = int(self.max_steps_var.get()) or 1
        max_iterations = 10

        prev_cents = initial_cents
        # tracking overshoots/reducciones
        iteration = 0

        for it in range(max_iterations):
            iteration += 1
            # calcular pasos sugeridos en base a prev_cents, pero no más que initial_n
            suggested = int(round(abs(prev_cents) / cents_per_step)) if cents_per_step > 0 else initial_n
            if suggested <= 0:
                steps = initial_n
            else:
                steps = min(initial_n, suggested)

            # si estamos en iteraciones posteriores y hubo overshoot, reducir según n/(iteration^2)
            if iteration > 1:
                reduced = max(1, int(round(initial_n / (iteration ** 2))))
                steps = min(steps, reduced)

            if steps <= 0:
                break

            direction = '+' if prev_cents < 0 else '-'

            ok = self._move_and_wait(direction, steps)
            if not ok:
                break  # <--- Corrige la indentación aquí

            # esperar a que update_loop publique latest_cents
            t0 = time.time()
            new_cents = getattr(self, "latest_cents", None)
            while new_cents is None and time.time() - t0 < 1.0:
                time.sleep(0.05)
                new_cents = getattr(self, "latest_cents", None)
            if new_cents is None:
                break

            # si sign flipped -> overshoot: revert parcialmente y contar iteración (ya se incrementó)
            if (prev_cents < 0 and new_cents > 0) or (prev_cents > 0 and new_cents < 0):
                # revert using reduced steps (n/(iteration^2)), al menos 1 paso
                reverse_steps = max(1, int(round(initial_n / (iteration ** 2))))
                rev_dir = '-' if direction == '+' else '+'
                self._move_and_wait(rev_dir, reverse_steps)
                time.sleep(0.08)

            # si ya afinada -> salir
            if abs(new_cents) <= GREEN_CENTS:
                break

            # actualizar prev_cents para siguiente iteración
            prev_cents = new_cents

    def _is_freq_stable(self, freq):
        """
        Determina si la frecuencia 'freq' se mantiene estable respecto al candidato actual.
        Uso de umbral en cents para evitar considerar micro-fluctuaciones.
        Retorna True si estable durante STABLE_MS_REQUIRED.
        """
        now = time.time() * 1000.0  # ms
        if self._stable_candidate_freq is None:
            # iniciar nuevo candidato
            self._stable_candidate_freq = freq
            self._stable_since = now
            return False
        # comparar en cents entre candidato y nueva medida
        cand = self._stable_candidate_freq
        c = cents_difference(freq, cand)
        if c is None or not np.isfinite(c):
            # reset candidato
            self._stable_candidate_freq = freq
            self._stable_since = now
            return False
        if abs(c) <= STABLE_CENTS_THRESHOLD:
            # sigue estable
            elapsed = now - (self._stable_since or now)
            return elapsed >= STABLE_MS_REQUIRED
        else:
            # cambió significativamente -> reiniciar candidato
            self._stable_candidate_freq = freq
            self._stable_since = now
            return False

    def update_loop(self):
        if not self.running:
            return
        try:
            audio = sd.rec(CHUNK, samplerate=FS, channels=1, dtype='float32', device=self.device_index)
            sd.wait()
            data = np.nan_to_num(audio.flatten())
        except Exception as e:
            self.note_label.config(text="Error", fg="red")
            self.freq_var.set(f"Error: {e}")
            self.root.after(UPDATE_MS, self.update_loop)
            return

        window = np.hanning(len(data))
        fft = np.fft.rfft(data * window)
        mag = np.abs(fft) / len(data)
        self.fft_data = mag
        self.line.set_ydata(self.fft_data)
        self.ax.set_ylim(0, max(1e-6, self.fft_data.max()*1.2))
        self.canvas.draw_idle()

        freq = get_freq_autocorr(data)
        if freq <= 0 or not np.isfinite(freq):
            self.note_label.config(text="—", fg="black")
            self.freq_var.set("Freq: - Hz")
            self.cents_var.set("Cents: -")
            self.root.after(UPDATE_MS, self.update_loop)
            return

        self.history.append(freq)
        freq_s = float(np.mean(self.history))
        self.latest_freq = freq_s

        if self.mode_var.get() == "Afinador guitarra":
            sel_string = self.string_var.get()
            target_freq = GUITAR_STRINGS.get(sel_string)
            cents = cents_difference(freq_s, target_freq)
            if cents is None or not np.isfinite(cents):
                cents = 0.0
            self.freq_var.set(f"Freq: {freq_s:.1f} Hz (obj: {target_freq:.1f} Hz)")
            self.cents_var.set(f"Cents: {cents:+.1f}")

            action = ""
            self.latest_cents = cents

            # Si la cuerda ya fue marcada como completada, NO mandar más comandos (hasta reiniciar o cambiar selección)
            if sel_string in self.completed_strings:
                action = "Afinada (pausada)"
                color = "green"
                note_name, octave, _ = freq_to_note_name(freq_s)
                self.note_label.config(text=f"{note_name}{octave}\n{action}", fg=color)
                self.root.after(UPDATE_MS, self.update_loop)
                return

            # Requerir estabilidad: solo tomar referencia si la frecuencia se ha mantenido estable > STABLE_MS_REQUIRED
            stable = self._is_freq_stable(freq_s)

            if not stable:
                # mostrar estado esperando estabilidad
                if abs(cents) <= GREEN_CENTS:
                    action = "Afinada (esperando estabilidad)"
                    color = "green"
                    self.completed_strings.add(sel_string)
                    self.update_completed_label()
                else:
                    action = "Esperando frecuencia estable"
                    color = "black"
                note_name, octave, _ = freq_to_note_name(freq_s)
                self.note_label.config(text=f"{note_name}{octave}\n{action}", fg=color)
                self.root.after(UPDATE_MS, self.update_loop)
                return

            # Si estable y fuera del rango naranja, iniciar afinado automático (thread)
            if abs(cents) > ORANGE_CENTS and self.motor_enabled_var.get() and self.motor:
                # evitar lanzar múltiples threads
                if not hasattr(self, "_tuning_thread") or not getattr(self, "_tuning_thread").is_alive():
                    cents_snapshot = cents
                    # iniciar algoritmo iterativo en hilo aparte
                    def tuning_task():
                        self.iterative_tune(cents_snapshot, np.sign(cents_snapshot))
                    self._tuning_thread = threading.Thread(target=tuning_task, daemon=True)
                    self._tuning_thread.start()
                if cents < -ORANGE_CENTS:
                    action = "Grave → tensar"
                else:
                    action = "Agudo → aflojar"
                color = "red"
            else:
                if abs(cents) <= GREEN_CENTS:
                    action = "Afinada"
                    color = "green"
                    self.completed_strings.add(sel_string)
                    self.update_completed_label()
                elif abs(cents) <= ORANGE_CENTS:
                    action = "Cerca"
                    color = "orange"
                else:
                    action = "Estable, sin acción"
                    color = "black"

            note_name, octave, _ = freq_to_note_name(freq_s)
            self.note_label.config(text=f"{note_name}{octave}\n{action}", fg=color)
            self.root.after(UPDATE_MS, self.update_loop)
            return

        # Normal mode
        note_name, octave, note_freq = freq_to_note_name(freq_s)
        cents_to_note = cents_difference(freq_s, note_freq) if note_freq else 0
        self.freq_var.set(f"Freq: {freq_s:.1f} Hz")
        self.cents_var.set(f"Cents: {cents_to_note:+.1f}")
        color = "green" if abs(cents_to_note) <= GREEN_CENTS else "orange" if abs(cents_to_note) <= ORANGE_CENTS else "black"
        self.note_label.config(text=f"{note_name}{octave}", fg=color)

        self.root.after(UPDATE_MS, self.update_loop)

    def open_advanced_options(self):
        # Ventana de opciones avanzadas
        win = tk.Toplevel(self.root)
        win.title("Opciones avanzadas")
        win.grab_set()
        frm = ttk.Frame(win, padding=12)
        frm.pack(fill='both', expand=True)

        # --- QUITA EL CAMPO DE DISPOSITIVO DE ENTRADA DE OPCIONES AVANZADAS ---
        row = 0
        # (No agregar el campo de dispositivo de entrada aquí)

        # --- Parámetros a mostrar/editar ---
        param_defs = [
            ("FS", "Frecuencia de muestreo", int, "Hz"),
            ("CHUNK", "Tamaño de bloque (CHUNK)", int, ""),
            ("UPDATE_MS", "Intervalo actualización", int, "ms"),
            ("SMOOTH_N", "Promedio frecuencias", int, ""),
            ("A4_FREQ", "A4 (La4)", float, "Hz"),
            ("ORANGE_CENTS", "Cents naranja", int, ""),
            ("GREEN_CENTS", "Cents verde", int, ""),
            ("STABLE_MS_REQUIRED", "Estabilidad requerida", int, "ms"),
            ("STABLE_CENTS_THRESHOLD", "Tolerancia estabilidad", float, "cents"),
        ]
        self.advanced_vars = {}
        for i, (key, label, typ, unit) in enumerate(param_defs):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky='w', padx=4, pady=2)
            val = globals().get(key, None)
            var = tk.StringVar(value=str(val))
            self.advanced_vars[key] = (var, typ)
            ttk.Entry(frm, textvariable=var, width=12).grid(row=i, column=1, padx=4)
            ttk.Label(frm, text=unit).grid(row=i, column=2, sticky='w')

        # --- Campos de calibración del motor ---
        row = len(param_defs)
        ttk.Label(frm, text="Cents/step (calibrar):").grid(row=row, column=0, sticky='w', padx=4, pady=2)
        cents_var = tk.StringVar(value=str(self.cents_per_step_var.get()))
        ttk.Entry(frm, textvariable=cents_var, width=12).grid(row=row, column=1, padx=4)
        ttk.Label(frm, text="").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(frm, text="Max steps/action (n):").grid(row=row, column=0, sticky='w', padx=4, pady=2)
        max_steps_var = tk.StringVar(value=str(self.max_steps_var.get()))
        ttk.Entry(frm, textvariable=max_steps_var, width=12).grid(row=row, column=1, padx=4)
        ttk.Label(frm, text="").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(frm, text="Step timeout (s):").grid(row=row, column=0, sticky='w', padx=4, pady=2)
        timeout_var = tk.StringVar(value=str(self.step_timeout_var.get()))
        ttk.Entry(frm, textvariable=timeout_var, width=12).grid(row=row, column=1, padx=4)
        ttk.Label(frm, text="").grid(row=row, column=2, sticky='w')

        row += 1
        motor_enabled_var = tk.BooleanVar(value=self.motor_enabled_var.get())
        ttk.Checkbutton(frm, text="Motor enabled", variable=motor_enabled_var).grid(row=row, column=0, sticky='w', padx=4, pady=2)

        def guardar():
            # Ya no se cambia el dispositivo de entrada aquí
            # Actualiza los parámetros globales
            for key, (var, typ) in self.advanced_vars.items():
                try:
                    val = typ(var.get())
                    globals()[key] = val
                    import main
                    setattr(main, key, val)
                except Exception:
                    pass
            # Actualiza los parámetros de calibración del motor
            try:
                self.cents_per_step_var.set(float(cents_var.get()))
            except Exception:
                pass
            try:
                self.max_steps_var.set(int(max_steps_var.get()))
            except Exception:
                pass
            try:
                self.step_timeout_var.set(float(timeout_var.get()))
            except Exception:
                pass
            self.motor_enabled_var.set(motor_enabled_var.get())
            win.destroy()

        btns = ttk.Frame(frm)
        btns.grid(row=row+1, column=0, columnspan=3, pady=10)
        ttk.Button(btns, text="Guardar", command=guardar).pack(side='left', padx=6)
        ttk.Button(btns, text="Cancelar", command=win.destroy).pack(side='left', padx=6)


