import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
original_image = cv2.imread('./Images/Fig0445(a) Characters Test Pattern 688x688.tif', cv2.IMREAD_GRAYSCALE)
ori_img = cv2.resize(original_image, (512, 512))

# Add Gaussian noise to the image
img = cv2.add(ori_img,np.random.normal(0, 0.5, ori_img.shape).astype(np.uint8))

D0 = 10

# Calculate the frequency domain representation
fimg = np.fft.fftshift(np.fft.fft2(img))
fimg1 = np.fft.fftshift(np.fft.fft2(ori_img))

(row, column) = img.shape
D=np.zeros((row,column))
for u in range(row):
        for v in range(column):            
            D[u,v]=np.sqrt( (u - row/2)**2 + (v - column/2)**2)
            

# Gaussian High-Pass Filter
ghf = 1 - np.exp(-((D**2) / (2 * D0**2)))
foutput_img = fimg * ghf
tmp_img = np.abs(np.fft.ifft2(foutput_img))
gaussian_hf_img =tmp_img/255



foutput_img_ori = fimg1 * ghf
tmp_img_ori = np.abs(np.fft.ifft2(foutput_img_ori))
gaussian_hf_img_ori = tmp_img_ori/255

# Ideal High-Pass Filter
idhf = D > D0
foutput_img = fimg * idhf
tmp_img = np.abs(np.fft.ifft2(foutput_img))
ideal_hf_img = tmp_img/255

idhf = D > D0
foutput_img_ori = fimg1 * idhf
tmp_img_ori = np.abs(np.fft.ifft2(foutput_img_ori))
ideal_hf_img_ori = tmp_img_ori/255

# Plot Images
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(ori_img, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(gaussian_hf_img_ori, cmap='gray')
plt.title('Gaussian High Pass Image')

plt.subplot(2, 3, 3)
plt.imshow(ideal_hf_img_ori, cmap='gray')
plt.title('Ideal High Pass Image')

plt.subplot(2, 3, 4)
plt.imshow(img, cmap='gray')
plt.title('Noisy Original Image')

plt.subplot(2, 3, 5)
plt.imshow(gaussian_hf_img, cmap='gray')
plt.title('Noisy Gaussian High Pass Image')

plt.subplot(2, 3, 6)
plt.imshow(ideal_hf_img, cmap='gray')
plt.title('Noisy Ideal High Pass Image')

plt.show()