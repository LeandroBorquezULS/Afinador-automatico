## Afinador Automático

Este proyecto es un **afinador automático** desarrollado con el objetivo principal de ofrecer una herramienta de fácil acceso y uso para músicos, estudiantes, y cualquier persona interesada en afinar instrumentos musicales de manera precisa y rápida.

## Objetivo

El objetivo de este código es brindar un afinador automático que cualquiera pueda utilizar, permitiendo la afinación de instrumentos musicales de forma sencilla, intuitiva y rápida, sin necesidad de conocimientos técnicos avanzados. La interfaz y la lógica han sido diseñadas para minimizar la complejidad y maximizar la accesibilidad.

## ¿Cómo funciona?

El proyecto combina **Python** (95.5%) y **C++** (4.5%) para aprovechar las ventajas de ambos lenguajes:

- **Python:** Se encarga mayormente del procesamiento de señales de audio, la interfaz de usuario y la lógica principal del afinador.
- **C++:** Se utiliza para funciones de procesamiento de audio que requieren alta eficiencia y velocidad, como la detección rápida de frecuencia fundamental.

El afinador recibe una señal de audio (por ejemplo, de un micrófono), procesa la señal para detectar la frecuencia fundamental con algoritmos avanzados, y muestra al usuario la nota detectada y qué tan cerca está de la afinación perfecta.

## Características principales

- **Fácil de usar:** Interfaz accesible y minimalista.
- **Automático:** No requiere ajustes manuales por parte del usuario.
- **Precisión:** Algoritmos robustos de análisis de frecuencia.
- **Multiplataforma:** Pensado para funcionar en diferentes sistemas operativos.
- **Código abierto:** Puedes revisar, modificar y mejorar el código libremente.

## Tecnologías utilizadas

- **Python**: Librerías para procesamiento de audio como `numpy`, `scipy`, y `sounddevice` o `pyaudio`.
- **C++**: Módulos optimizados integrados sobre Python para análisis eficiente de la señal.

## Instalación y uso

1. Clona/Descargar este repositorio:
2. Instala las librerias requeridas.
3. Ejecuta el script principal que se encuentra en "main" siguiendo las instrucciones del repositorio.

## Materiales
(estos son los materiales esenciales para su funcionamiento)
- Motor Paso a Paso 28BYJ-48, 5v
- Driver ULN2003
- ESP32 D1 mini

## Contribuciones

¡Cualquier mejora, sugerencia o reporte es bienvenido! Este proyecto busca crecer como una herramienta comunitaria de acceso abierto.

---

**Hecho por [Leandro Borquez-Cristián Contreras-Maximiliano Molina-Jose Barraza-Maximiliano Oyarse-Piero Araya]**
-Bajo el contexto de la aplicación de Series de Fourier, en el marco del ramo Matemáticas IV, impartido por el Doctor Eric Roberto Jeltsch Figueroa en la Universidad de La Serena.
-Proyecto enfocado en el acceso universal a herramientas musicales inteligentes.
