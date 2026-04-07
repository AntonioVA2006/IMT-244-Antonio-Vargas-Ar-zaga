import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

y, fs = librosa.load('audio_con_ruido.wav')
y_fft=np.fft.fft(y)
n=len(y)
T= 1.0/fs
frecuencias = np.fft.fftfreq(n, T)
plt.figure(figsize=(12,8))

plt.subplot(2, 2, 1)
plt.plot(frecuencias[:n//2], np.abs(y_fft)[:n//2])
plt.title("Espectro Original")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)

frecuencia_corte = 1000
Y_filtrado = y_fft.copy() 
 
mask = (np.abs(frecuencias) > frecuencia_corte) 
Y_filtrado[mask] = 0

plt.subplot(2, 2, 2)
plt.plot(frecuencias[:n//2], np.abs(Y_filtrado)[:n//2])
plt.title("Espectro Filtrado (Pasa-bajo)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)

y_limpio = np.fft.ifft(Y_filtrado)
y_limpio = np.real(y_limpio)  
sf.write('audio_pasa_bajo.wav', y_limpio, fs)