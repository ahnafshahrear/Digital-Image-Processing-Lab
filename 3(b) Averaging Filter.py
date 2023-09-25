import cv2
import matplotlib.pyplot as plt
import numpy as np

#... Function for Image Plot 
def plot_image(image, text, subplot):
    plt.subplot(2, 3, subplot)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(text)

#... Function for applying Averaging Filter
def apply_average_filter(image, mask):
    height, width = image.shape
    average_image = []
    x = mask // 2
    for c in range(height):
        new_row = []
        for r in range(width):
            pixel = 0
            for i in range(-x, x + 1, 1):
                for j in range(-x, x + 1, 1):
                    if (c + i >= 0 and c + i < height and r + j >= 0 and r + j < width):
                        pixel += image[c + i, r + j] // (mask * mask)
            new_row.append(pixel)
        average_image.append(new_row)
    return np.uint8(average_image)

#... Function for calculating Peak Signal to Noise Ratio (PSNR)
def psnr(image1, image2):
    mse = np.mean((image1 - image2)**2)
    psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    return round(psnr, 2)

#... Function for applying Salt & Pepper Noise
def salt_pepper_noise(image, amount):
    noisy_image = image.copy()
    for k in range(amount):
        index = []
        for i in range(1, 5, 1):
            index.append(np.random.randint(0, image.shape[0]))
        noisy_image[index[0], index[1]], noisy_image[index[2], index[3]] = 0, 255
    return noisy_image

#... Importing & plotting Original Image
original_image = cv2.resize(cv2.imread('Joker.jpg', 0), (256, 256))
plt.figure(figsize = (13, 7))
plot_image(original_image, "Original Image", 1)

#... Applying noise
noisy_image = salt_pepper_noise(original_image, 1000)
plot_image(noisy_image, "Noisy Image", 2)

#... Applying Averaging Filter
mask = 3
for k in range(3, 7, 1):
    avg_image = apply_average_filter(original_image, mask)
    avg_psnr = psnr(original_image, avg_image)
    plot_image(avg_image, f"{mask}x{mask} Mask and PSNR = {avg_psnr}", k)
    mask += 2

plt.show()