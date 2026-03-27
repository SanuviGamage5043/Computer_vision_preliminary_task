import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Image_4.jpg')  

if img is None:
    print("Error: Image not found!")
    exit()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(img_gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite('Q8_threshold.png', thresh)

kernel = np.ones((5,5), np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('Q8_cleaned.png', closing)

contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(img_gray)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:  
        cv2.drawContours(mask, [cnt], -1, 255, -1)

cv2.imwrite('Q8_mask.png', mask)

segmented = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite('Q8_segmented.png', segmented)

# ------------------ Display ------------------
plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original CT Image")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(opening, cmap='gray')
plt.title("Opening")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(closing, cmap='gray')
plt.title("Closing")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(mask, cmap='gray')
plt.title("Organ Mask")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title("Segmented Organs")
plt.axis('off')

plt.tight_layout()
plt.show()