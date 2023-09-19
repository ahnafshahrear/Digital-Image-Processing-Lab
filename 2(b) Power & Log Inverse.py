import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_image(image, text, subplot):
    plt.subplot(2, 2, subplot)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(text)

original_image = cv2.resize(cv2.imread('Joker.jpg', 0), (512, 512))
plot_image(original_image, "Original Image", 1)

height, width = original_image.shape
power_image = []

for c in range(height):
    image_row = []
    for r in range(width):
        pixel = (original_image[c, r] / 255.0) ** 1.5
        image_row.append(pixel * 255)
    power_image.append(image_row)

plot_image(np.uint8(power_image), "Power Transformed Image", 2)

inverse_log_image = []

for c in range(height):
    image_row = []
    for r in range(width):
        pixel = 10 ** ((original_image[c, r] / 255.0) - 1)
        image_row.append(pixel * 255)
    inverse_log_image.append(image_row)

plot_image(np.uint8(inverse_log_image), "Inverse Log Transformed Image", 3)

difference_image = cv2.absdiff(np.uint8(power_image), np.uint8(inverse_log_image))
plot_image(difference_image, "Difference Image", 4)

plt.show()