import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

img = cv2.imread('Image_3.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not found!")
    exit()

# Salt & Pepper Noise Function
def add_sp_noise(image, amount=0.1):
    noisy = image.copy()
    num = int(amount * image.size)
    coords = [np.random.randint(0, i, num) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, num) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy
sp = add_sp_noise(img, 0.1)
cv2.imwrite('Q6_sp_noise.png', sp)

# Laplacian of the original image
lap = cv2.Laplacian(img, cv2.CV_64F)
lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
lap_norm = lap_norm.astype(np.uint8)
I_prime = cv2.add(img, sp)
I_prime = cv2.add(I_prime, lap_norm)
cv2.imwrite('Q6_I_prime.png', I_prime)

# Wavelet Decomposition
coeffs2 = pywt.dwt2(I_prime, 'haar')
LL, (LH, HL, HH) = coeffs2

LH[:] = 0
HL[:] = 0
HH[:] = 0

reconstructed = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX)
reconstructed = reconstructed.astype(np.uint8)
cv2.imwrite('Q6_reconstructed.png', reconstructed)

# ------------------ Display ------------------
plt.figure(figsize=(12,6))

# Original Image
plt.subplot(1,4,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Salt & Pepper Noise
plt.subplot(1,4,2)
plt.imshow(sp, cmap='gray')
plt.title("Salt & Pepper (10%)")
plt.axis('off')

# I'
plt.subplot(1,4,3)
plt.imshow(I_prime, cmap='gray')
plt.title("I' = I + SP + L(I)")
plt.axis('off')

# Reconstructed
plt.subplot(1,4,4)
plt.imshow(reconstructed, cmap='gray')
plt.title("Reconstructed (Smooth)")
plt.axis('off')

plt.tight_layout()
plt.show()