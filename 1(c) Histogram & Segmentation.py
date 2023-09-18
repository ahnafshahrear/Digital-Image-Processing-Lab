import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_image(image, text, subplot):
    plt.subplot(2, 2, subplot)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(text)

def histogram(image, subplot):
    height, width = original_image.shape
    histogram = np.zeros(256)
    
    for c in range(height):
        for r in range(width):
            histogram[image[c, r]] += 1

    plt.subplot(2, 2, subplot)
    plt.bar(range(256), histogram, width = 1.0, color = "gray")

original_image = cv2.imread("Joker.jpg", 0)
original_image = cv2.resize(original_image, (512, 512))

plot_image(original_image, "Original Image", 1)

histogram(original_image, 2)

segmented_image = original_image
height, width = original_image.shape

for c in range(height):
    for r in range(width):
        segmented_image[c, r] = 0 if original_image[c, r] < 128 else 255

plot_image(segmented_image, "Single Threshold Segmented", 3)
histogram(segmented_image, 4)

plt.show()