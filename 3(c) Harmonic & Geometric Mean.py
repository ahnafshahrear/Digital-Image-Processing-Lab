import cv2
import matplotlib.pyplot as plt
import numpy as np

#... Function for Image Plot 
def plot_image(image, text, subplot):
    plt.subplot(2, 2, subplot)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(text)

def apply_geometric_mean(image, mask):
    height, width = image.shape
    average_image = []
    x = mask // 2
    for c in range(height):
        new_row = []
        for r in range(width):
            pixel, count = 1, 0
            for i in range(-x, x + 1, 1):
                for j in range(-x, x + 1, 1):
                    if (c + i >= 0 and c + i < height and r + j >= 0 and r + j < width):
                        if (image[c + i, r + j] != 0):
                            count += 1
                            pixel = pixel * int(image[c + i, r + j])
            count = 1 if count == 0 else count
            new_row.append(pixel**(1/count))
        average_image.append(new_row)
    return np.uint8(average_image)

def apply_harmonic_mean(image, mask):
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
                        pixel = pixel + float(1.0 / (image[c + i, r + j] + 1))
            pixel = (mask * mask) / pixel
            pixel = 255 if pixel > 255 else pixel
            new_row.append(pixel)
        average_image.append(new_row)
    return np.uint8(average_image)

#... Function for calculating Peak Signal to Noise Ratio (PSNR)
def psnr(image1, image2):
    image1 = np.array(image1, dtype=np.float64)
    image2 = np.array(image2, dtype=np.float64)
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
noisy_image = salt_pepper_noise(original_image, 10000)
plot_image(noisy_image, f"Noisy Image with PSNR = {psnr(original_image, noisy_image)}", 2)
print(psnr(original_image, noisy_image))

geometric_image = apply_geometric_mean(noisy_image, 3)
geo_psnr = psnr(original_image, geometric_image)
plot_image(geometric_image, f"Geometric Mean Filter {geo_psnr}", 3)

harmonic_image = apply_harmonic_mean(noisy_image, 3)
harmo_psnr = psnr(original_image, harmonic_image)
plot_image(harmonic_image, f"Harmonic Mean Filter {harmo_psnr}", 4)

plt.show()