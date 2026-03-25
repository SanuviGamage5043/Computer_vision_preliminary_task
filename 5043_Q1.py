import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('Image_1.jpg')

if img is None:
    print("Image not found!")
    exit()

# Convert color
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Kernel sizes
kernels = [3, 5, 11, 15]

# Plot results
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis('off')

for i, k in enumerate(kernels):
    blurred = cv2.blur(img_rgb, (k, k))

    # Save image
    cv2.imwrite(f'avg_{k}.png', cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))

    plt.subplot(2,3,i+2)
    plt.imshow(blurred)
    plt.title(f"{k}x{k}")
    plt.axis('off')

plt.tight_layout()
plt.show()