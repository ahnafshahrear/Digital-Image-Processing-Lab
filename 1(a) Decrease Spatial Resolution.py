import cv2
import matplotlib.pyplot as plt
import numpy as np

original_image = cv2.imread("Joker.jpg", 0)
original_image = cv2.resize(original_image, (512, 512))

for k in range(1, 9):
    height, width = original_image.shape
    plt.subplot(2, 4, k)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title(f"{height}x{width}")
    decreased_image = []
    for c in range(0, height, 2):
        new_row = []
        for r in range(0, width, 2):
            pixel = (original_image[c, r] * 0.25) + (original_image[c + 1, r] * 0.25)
            pixel = pixel + (original_image[c, r + 1] * 0.25) + (original_image[c + 1, r + 1] * 0.25)
            new_row.append(pixel)
        decreased_image.append(new_row)
    decreased_image = np.uint8(decreased_image)
    original_image = decreased_image

plt.show()