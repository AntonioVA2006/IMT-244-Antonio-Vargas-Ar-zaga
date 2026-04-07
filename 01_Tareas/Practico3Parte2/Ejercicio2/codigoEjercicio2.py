import numpy as np
import matplotlib.pyplot as plt
import librosa

# Cargar archivo de audio
audio, fs = librosa.load('audio_con_ruido.wav', sr=None)  # sr=None mantiene la frecuencia original

# Generar ruido blanco
ruido = np.random.normal(0, 0.1, len(audio))
audio_con_ruido = audio + ruido

# Calcular FFT
mitad = len(audio) // 2
fft_original = np.fft.fft(audio)
fft_con_ruido = np.fft.fft(audio_con_ruido)

amplitud = np.abs(fft_original[:mitad])
amplitud_ruido = np.abs(fft_con_ruido[:mitad])

# Calcular frecuencias
frecs_pos = np.linspace(0, fs/2, mitad)

# Comparar espectros
plt.figure(figsize=(12, 6))
plt.plot(frecs_pos, amplitud, alpha=0.7, label='Original')
plt.plot(frecs_pos, amplitud_ruido, alpha=0.7, label='Con ruido')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.legend()
plt.xlim(0, fs/2)
plt.grid()
plt.show()