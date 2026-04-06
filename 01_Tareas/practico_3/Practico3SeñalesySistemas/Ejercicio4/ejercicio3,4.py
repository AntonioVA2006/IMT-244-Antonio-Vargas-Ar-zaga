import numpy as np
import matplotlib.pyplot as plt


fs = 510
t = np.linspace(0, 1, fs, endpoint=False)


x = np.sin(2 * np.pi * 250 * t) + np.sin(2 * np.pi * 1000 * t)              


N = len(x)
X_fft = np.fft.fft(x)                      
frecuencias = np.fft.fftfreq(N, 1/fs)      


n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(X_fft[:n_mitad]) * 2 / N 


plt.figure(figsize=(10, 12))

plt.subplot(2,1,1)
plt.plot(t, x, color='purple')
plt.title("Señal compuesta (Suma en el Tiempo)")
plt.grid(True)


plt.subplot(2,1,2)
#plt.stem(f_positivas, magnitud, linefmt='green', markerfmt='go', basefmt=" ")
plt.plot(f_positivas, magnitud, color='green', linewidth=2)
plt.title("Espectro de Frecuencias (FFT)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.xlim(0, 300)  
plt.grid(True)

plt.tight_layout()
plt.show()