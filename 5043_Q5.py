import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('Image_1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Image_2.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('Image_3.jpg', cv2.IMREAD_GRAYSCALE)

# Q1: Average Filter

kernel = np.ones((15,15), np.float32) / (15*15)
A1 = cv2.filter2D(img1, -1, kernel)
B1 = cv2.blur(img1, (15,15))

D1 = cv2.absdiff(A1, B1)

cv2.imwrite('Q5_avg_A.png', A1)
cv2.imwrite('Q5_avg_B.png', B1)
cv2.imwrite('Q5_avg_diff.png', D1)


# Q2: Median Filter

def add_sp_noise(image, amount=0.2):
    noisy = image.copy()
    num = int(amount * image.size)
    
    coords = [np.random.randint(0, i, num) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    
    coords = [np.random.randint(0, i, num) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    
    return noisy

noisy_img = add_sp_noise(img2, 0.2)

A2 = cv2.medianBlur(noisy_img, 11)
B2 = cv2.medianBlur(noisy_img, 11)

D2 = cv2.absdiff(A2, B2)

cv2.imwrite('Q5_median_A.png', A2)
cv2.imwrite('Q5_median_B.png', B2)
cv2.imwrite('Q5_median_diff.png', D2)


# Q3: Gaussian Filter

def gaussian_kernel(size, sigma):
    k = cv2.getGaussianKernel(size, sigma)
    return k @ k.T

g_kernel = gaussian_kernel(15, 1)
A3 = cv2.filter2D(img3, -1, g_kernel)
B3 = cv2.GaussianBlur(img3, (15,15), 1)

D3 = cv2.absdiff(A3, B3)

cv2.imwrite('Q5_gaussian_A.png', A3)
cv2.imwrite('Q5_gaussian_B.png', B3)
cv2.imwrite('Q5_gaussian_diff.png', D3)


# Display for all

plt.figure(figsize=(12,10))

# Q1
titles = ['A', 'B', 'A-B']
images = [A1, B1, D1]

for i in range(3):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f'Q1 {titles[i]}')
    plt.axis('off')

# Q2
images = [A2, B2, D2]

for i in range(3):
    plt.subplot(3,3,i+4)
    plt.imshow(images[i], cmap='gray')
    plt.title(f'Q2 {titles[i]}')
    plt.axis('off')

# Q3
images = [A3, B3, D3]

for i in range(3):
    plt.subplot(3,3,i+7)
    plt.imshow(images[i], cmap='gray')
    plt.title(f'Q3 {titles[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()
