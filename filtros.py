"""Compara filtros FIR e IIR usando una señal simulada."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Señal de entrada reproducible: tres tonos y ruido blanco.
fs = 1000
t = np.arange(2 * fs) / fs
tonos = np.array([20, 100, 250])
ruido = np.random.default_rng(42).normal(0, 0.3, t.size)
entrada = np.sin(2 * np.pi * tonos[:, None] * t).sum(axis=0) + ruido
frecuencias = np.fft.rfftfreq(t.size, 1 / fs)
salida = Path(__file__).resolve().parent

def espectro(x):
    """Magnitud unilateral de la FFT, normalizada por la longitud."""
    magnitud = np.abs(np.fft.rfft(x)) / x.size
    magnitud[1:-1] *= 2
    return magnitud

def decibelios(h):
    return 20 * np.log10(np.maximum(np.abs(h), 1e-8))

# Un diseño FIR de ventana Hamming y un Butterworth IIR por cada banda.
configuraciones = [
    ("Pasa bajas", "lowpass", 50),
    ("Pasa altas", "highpass", 180),
    ("Pasa bandas", "bandpass", [70, 140]),
]
fig, ejes = plt.subplots(3, 3, figsize=(12, 8), layout="constrained")
for (nombre, modo, corte), (ax_t, ax_f, ax_h) in zip(configuraciones, ejes):
    fir = signal.firwin(101, corte, window="hamming", pass_zero=modo, fs=fs)
    iir = signal.butter(4, corte, btype=modo, fs=fs, output="sos")
    # Filtrado causal. En el tiempo se incluye el desfase de cada diseño.
    y_fir = signal.lfilter(fir, [1], entrada)
    y_iir = signal.sosfilt(iir, entrada)
    ax_t.plot(t, entrada, color="0.65", label="Entrada")
    ax_f.plot(frecuencias, espectro(entrada), color="0.65", label="Entrada")
    for etiqueta, y, color in [("FIR", y_fir, "#0072B2"), ("IIR", y_iir, "#D55E00")]:
        assert np.isfinite(y).all()
        ax_t.plot(t, y, label=etiqueta, color=color, linewidth=1)
        ax_f.plot(frecuencias, espectro(y), label=etiqueta, color=color, linewidth=1)
    f, h_fir = signal.freqz(fir, worN=2048, fs=fs)
    _, h_iir = signal.sosfreqz(iir, worN=f, fs=fs)
    ax_h.plot(f, decibelios(h_fir), label="FIR", color="#0072B2")
    ax_h.plot(f, decibelios(h_iir), label="IIR", color="#D55E00")
    for limite in np.atleast_1d(corte):
        ax_h.axvline(limite, color="0.5", linestyle=":", linewidth=1)
    ax_t.set(xlim=(0.25, 0.40), xlabel="Tiempo (s)", ylabel=nombre)
    ax_f.set(xlim=(0, 300), ylim=(0, 1.15), xlabel="Frecuencia (Hz)")
    ax_h.set(xlim=(0, 300), ylim=(-100, 5), xlabel="Frecuencia (Hz)")
    for ax in (ax_t, ax_f, ax_h):
        ax.grid(alpha=0.22)
    print(nombre)
    print("FIR (dB):", decibelios(signal.freqz(fir, worN=tonos, fs=fs)[1]).round(2))
    print("IIR (dB):", decibelios(signal.sosfreqz(iir, worN=tonos, fs=fs)[1]).round(2))

for ax, titulo in zip(ejes[0], ["Señal en el tiempo", "Espectro de amplitud", "Respuesta en frecuencia (dB)"]):
    ax.set_title(titulo)
    ax.legend(loc="best", fontsize=8)
fig.savefig(salida / "comparacion_filtros.png", dpi=180)
