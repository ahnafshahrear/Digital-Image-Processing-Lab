import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_image(image, text, subplot):
    plt.subplot(1, 3, subplot)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(text)

image = cv2.resize(cv2.imread('Joker.jpg', cv2.IMREAD_GRAYSCALE), (512, 512))
plot_image(image, "Original Image", 1)

three_bit_image = (image >> 5) << 5
plot_image(three_bit_image, "Image using last 3 bits", 2)


difference = cv2.absdiff(np.uint8(image), np.uint8(three_bit_image))
plot_image(difference, "Difference_image", 3)

plt.show()