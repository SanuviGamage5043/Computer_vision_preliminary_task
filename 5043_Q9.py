import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Image_5.jpg', cv2.IMREAD_GRAYSCALE)  # Load MRI as grayscale
if img is None:
    print("Error: Image not found!")
    exit()

hist_eq = cv2.equalizeHist(img)
cv2.imwrite('Q9_hist_eq.png', hist_eq)

min_val = np.min(img)
max_val = np.max(img)
contrast_stretch = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
cv2.imwrite('Q9_contrast_stretch.png', contrast_stretch)

gaussian_filtered = cv2.GaussianBlur(img, (5,5), 1)  
cv2.imwrite('Q9_gaussian.png', gaussian_filtered)

titles = ['Original MRI', 'Histogram Equalization', 'Contrast Stretching', 'Gaussian Filter']
images = [img, hist_eq, contrast_stretch, gaussian_filtered]

plt.figure(figsize=(12,6))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()