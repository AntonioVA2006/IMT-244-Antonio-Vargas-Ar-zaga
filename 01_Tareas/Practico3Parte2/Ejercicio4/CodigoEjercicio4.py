import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('CuldeSac.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = rows//2, cols//2  

mask = np.ones((rows, cols), np.uint8)
r = 30 
cv2.circle(mask, (ccol, crow), r, 0, -1) 

fshift_filtered = fshift * mask

f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(mask, cmap='gray'), plt.title('Máscara (LPF)')
plt.subplot(133), plt.imshow(img_back, cmap='gray'), plt.title('Imagen Reconstruida')
plt.show()