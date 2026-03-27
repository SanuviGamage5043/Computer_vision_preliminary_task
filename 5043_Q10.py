import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

img = cv2.imread('Image_6.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not found!")
    exit()
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('Q10_binary.png', binary)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))  # 5x5 rectangle

erosion = cv2.erode(binary, kernel, iterations=1)
cv2.imwrite('Q10_erosion.png', erosion)

dilation = cv2.dilate(binary, kernel, iterations=1)
cv2.imwrite('Q10_dilation.png', dilation)

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cv2.imwrite('Q10_opening.png', opening)

closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('Q10_closing.png', closing)

labels = measure.label(binary, connectivity=2)
props = measure.regionprops(labels)

print("Basic Morphological Features:")
for i, prop in enumerate(props):
    print(f"Object {i+1}: Area = {prop.area}, Perimeter = {prop.perimeter}")

titles = ['Original', 'Binary', 'Erosion', 'Dilation', 'Opening', 'Closing']
images = [img, binary, erosion, dilation, opening, closing]

plt.figure(figsize=(14,6))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()