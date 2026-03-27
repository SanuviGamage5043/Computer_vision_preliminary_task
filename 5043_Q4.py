import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Image_3.jpg')
if img is None:
    print("Error: Image not found!")
    exit()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3 levels of Gaussian Pyramid
gp = [img_rgb]

for i in range(3):
    down = cv2.pyrDown(gp[i])
    gp.append(down)
    
for i, g in enumerate(gp):
    cv2.imwrite(f'Q4_gaussian_level_{i}.png',
                cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
plt.figure(figsize=(12,6))

for i in range(len(gp)):
    plt.subplot(2,2,i+1)
    plt.imshow(gp[i])
    plt.title(f'Gaussian Level {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 3 levels of Laplacian Pyramid
lp = []
for i in range(3, 0, -1):
    up = cv2.pyrUp(gp[i])
    up = cv2.resize(up, (gp[i-1].shape[1], gp[i-1].shape[0]))
    lap = gp[i-1].astype(float) - up.astype(float)
    lap_display = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
    lap_display = lap_display.astype(np.uint8)
    lp.append(lap_display)
last = cv2.normalize(gp[3], None, 0, 255, cv2.NORM_MINMAX)
last = last.astype(np.uint8)
lp.append(last)
for i, l in enumerate(lp):
    cv2.imwrite(f'Q4_laplacian_level_{i}.png',
                cv2.cvtColor(l, cv2.COLOR_RGB2BGR))
plt.figure(figsize=(12,6))
for i in range(len(lp)):
    plt.subplot(2,2,i+1)
    plt.imshow(lp[i])
    plt.title(f'Laplacian Level {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()