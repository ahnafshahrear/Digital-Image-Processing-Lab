import cv2
import matplotlib.pyplot as plt
import numpy as np

original_image = cv2.imread('Joker.jpg', 0)
original_image = cv2.resize(original_image, (0, 0), fx = 0.2, fy = 0.2)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
plt.title("Original Image")

intensity_start, intensity_end = 10, 100
enhanced_image = []
height, width = original_image.shape

for c in range(height):
    new_row = []
    for r in range(width):
        pixel = original_image[c, r]
        if (original_image[c, r] > intensity_start) and (original_image[c, r] < intensity_end):
            pixel = (1 + original_image[c, r]) ** 1.123
        pixel = 255 if pixel > 255 else pixel
        new_row.append(pixel)
    enhanced_image.append(new_row)

enhanced_image = np.uint8(enhanced_image)

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
plt.title("Enhanced Image")

plt.show()