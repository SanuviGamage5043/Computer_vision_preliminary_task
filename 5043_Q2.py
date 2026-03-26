import cv2
import numpy as np
import matplotlib.pyplot as plt

# a) salt & pepper noise function
def add_salt_pepper_noise(image, amount):
    noisy_img = image.copy()
    total_pixels = image.size
    num_salt = np.ceil(amount * total_pixels * 0.5).astype(int)
    num_pepper = np.ceil(amount * total_pixels * 0.5).astype(int)
    
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_img[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_img[coords[0], coords[1]] = 0
    return noisy_img

img = cv2.imread('Image_2.jpg', cv2.IMREAD_GRAYSCALE)

sp_10 = add_salt_pepper_noise(img, 0.10)
sp_20 = add_salt_pepper_noise(img, 0.20)

cv2.imwrite('Q2_sp_10.png', sp_10)
cv2.imwrite('Q2_sp_20.png', sp_20)

# b) median filter function
kernel_sizes = [3, 5, 11]

def apply_median_filter(noisy_img, kernel_sizes, prefix):
    plt.figure(figsize=(12,4))
    plt.subplot(1, len(kernel_sizes)+1, 1)
    plt.imshow(noisy_img, cmap='gray')
    plt.title(prefix)
    plt.axis('off')
    
    for i, k in enumerate(kernel_sizes):
        filtered = cv2.medianBlur(noisy_img, k)
        cv2.imwrite(f'Q2_{prefix}_median_{k}.png', filtered)
        plt.subplot(1, len(kernel_sizes)+1, i+2)
        plt.imshow(filtered, cmap='gray')
        plt.title(f'Median {k}x{k}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

apply_median_filter(sp_10, kernel_sizes, 'sp10')
apply_median_filter(sp_20, kernel_sizes, 'sp20')