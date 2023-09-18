import cv2
import matplotlib.pyplot as plt
import numpy as np

original_image = cv2.imread("Joker.jpg", 0)
original_image = cv2.resize(original_image, (512, 512))

height, width = original_image.shape

for k in range(1, 9):
    decreased_image = []
    level = 2**k
    step = 255 / (level - 1)
    for c in range(height):
        new_row = []
        for r in range(width):
            new_row.append(round(original_image[c, r] / step) * step)
        decreased_image.append(new_row)
    decreased_image = np.uint8(decreased_image)
    plt.subplot(2, 4, k)
    plt.imshow(cv2.cvtColor(decreased_image, cv2.COLOR_BGR2RGB))
    plt.title(f"{k} bit(s)")

plt.show()