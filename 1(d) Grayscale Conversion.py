import cv2
import matplotlib.pyplot as plt
import numpy as np

original_image = cv2.imread("Joker.jpg")
original_image = cv2.resize(original_image, (0, 0), fx = 0.2, fy = 0.2)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

grayscale_image = []
height, width = original_image.shape[:2]

for c in range(height):
    new_row = []
    for r in range(width):
        pixel = (original_image[c, r, 0] * .114) + (original_image[c, r, 1] * .587) + (original_image[c, r, 2] * .299)
        new_row.append(pixel)
    grayscale_image.append(new_row)

grayscale_image = np.uint8(grayscale_image)

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2RGB))
plt.title("Grayscale Image")

plt.show()