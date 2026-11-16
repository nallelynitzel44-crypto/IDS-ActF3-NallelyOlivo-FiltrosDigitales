# Implementación y evaluación de filtros digitales

El programa compara filtros pasa bajas, pasa altas y pasa bandas en versiones FIR de ventana Hamming e IIR Butterworth. Utiliza una señal simulada con componentes de 20, 100 y 250 Hz más ruido blanco.

## Ejecución

```bash
python3 -m pip install numpy scipy matplotlib
python3 filtros.py
```

El programa guarda `comparacion_filtros.png` junto al código e imprime las ganancias en dB para 20, 100 y 250 Hz, en ese orden. Las filas de la figura corresponden a los tres tipos de filtro. Sus columnas muestran la señal en el tiempo, el espectro de amplitud y la respuesta en frecuencia.
