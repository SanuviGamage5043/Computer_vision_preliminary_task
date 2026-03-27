import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

cover = cv2.imread('Image_3.jpg', cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread('watermark.png', cv2.IMREAD_GRAYSCALE)

if cover is None or watermark is None:
    print("Error: Image not found!")
    exit()

watermark = cv2.resize(watermark, (cover.shape[1]//2, cover.shape[0]//2))

coeffs = pywt.dwt2(cover, 'haar')
LL, (LH, HL, HH) = coeffs

alpha = 0.1  

LL_watermarked = LL + alpha * watermark


watermarked_img = pywt.idwt2((LL_watermarked, (LH, HL, HH)), 'haar')


watermarked_img = cv2.normalize(watermarked_img, None, 0, 255, cv2.NORM_MINMAX)
watermarked_img = watermarked_img.astype(np.uint8)

cv2.imwrite('Q7_watermarked.png', watermarked_img)

coeffs_w = pywt.dwt2(watermarked_img, 'haar')
LL_w, (_, _, _) = coeffs_w

extracted = (LL_w - LL) / alpha

extracted = cv2.normalize(extracted, None, 0, 255, cv2.NORM_MINMAX)
extracted = extracted.astype(np.uint8)

cv2.imwrite('Q7_extracted.png', extracted)

# ------------------ Display ------------------
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(cover, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(watermark, cmap='gray')
plt.title("Watermark")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(watermarked_img, cmap='gray')
plt.title("Watermarked")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(extracted, cmap='gray')
plt.title("Extracted")
plt.axis('off')

plt.tight_layout()
plt.show()