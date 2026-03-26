import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Image_3.jpg')

if img is None:
    print("Error: Image not found!")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel_sizes = [3, 5, 11, 15]

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

for i, k in enumerate(kernel_sizes):
    blurred = cv2.GaussianBlur(img_rgb, (k, k), 1)
    cv2.imwrite(f'Q3_gaussian_{k}.png', cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
    plt.subplot(2,3,i+2)
    plt.imshow(blurred)
    plt.title(f"{k}x{k} Kernel")
    plt.axis('off')

plt.tight_layout()
plt.show()