import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

y, fs = librosa.load('audio_con_ruido.wav')
y_fft=np.fft.fft(y)
n=len(y)
T= 1.0/fs
frecuencias = np.fft.fftfreq(n, T)

magnitudes = np.abs(y_fft)
magnitudes_positivas = magnitudes[:n//2]
frecuencias_positivas = frecuencias[:n//2]

idx_dominante = np.argmax(magnitudes_positivas)
frecuencia_dominante = frecuencias_positivas[idx_dominante]
amplitud_dominante = magnitudes_positivas[idx_dominante]

print(f"Frecuencia dominante: {frecuencia_dominante:.2f} Hz")
print(f"Amplitud en esa frecuencia: {amplitud_dominante:.2f}")

plt.plot(frecuencias[:n//2], np.abs(y_fft)[:n//2])

# plt.plot(frecuencias, np.abs(y_fft))
plt.title("Espectro de Frecuencia")
plt.xlabel("Frecuencia (Hz)")
plt.show()

Y_filtrado = y_fft.copy()
umbral_frecuencia = 1000 
ancho_banda = 50 
mask = (np.abs(frecuencias) > umbral_frecuencia - ancho_banda) & \
       (np.abs(frecuencias) < umbral_frecuencia + ancho_banda)
Y_filtrado[mask] = 0


y_limpio = np.fft.ifft(Y_filtrado)
y_limpio = np.real(y_limpio)  
sf.write('audio_limpioTarea1.wav', y_limpio, fs)