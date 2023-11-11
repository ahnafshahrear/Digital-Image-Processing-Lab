import cv2
import numpy as np
import matplotlib.pyplot as plt

def plotimg(image,x,y,z,st):
    plt.subplot(x,y,z)
    plt.imshow(image,cmap="gray")
    plt.title(st)

def gaussian(D0,f_img):
    M, N = f_img.shape
    Gaussian = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N): 
            D =np.sqrt( (u - M/2)**2 + (v - N/2)**2)
            Gaussian[u, v] = np.exp(-((D**2) / (2 * D0**2)))
    filtered_image=Gaussian*f_img
    
    return filtered_image
def butterworth(D0,n,f_img):
    M, N = f_img.shape
    H = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            H[u, v] = 1 / (1 + (D / D0)**(2 * n))
    G = f_img * H
    return G  
def main():
    # Load the grayscale image
    Original_image = cv2.imread('./Images/Fig0445(a) Characters Test Pattern 688x688.tif', cv2.IMREAD_GRAYSCALE)

    # Generate Gaussian noise
    noise = np.random.normal(7, 10, Original_image.shape).astype(np.uint8)

    # Add noise to the original image
    image = cv2.add(Original_image, noise)

    # Perform FFT on the image
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)  # Apply log for visualization
    plotimg(Original_image,3,2,1,"Original image")
    plotimg(image,3,2,2,"Noisy Image")
    plotimg(magnitude_spectrum,3,2,3,"DFT magnitude spectrum")

    #using butter worth filter and gaussian
    filtered_gaussian = gaussian(50,fft_shifted)
    filtered_butter = butterworth(15,2,fft_shifted)
    #perform Inverse fft on gaussian filtered image
    reconstructed_ishifted = np.fft.ifftshift(filtered_gaussian)
    reconstructed_ishifted_ifft = np.fft.ifft2(reconstructed_ishifted).real
    plotimg(reconstructed_ishifted_ifft,3,2,4,"Reconstructed image using Gaussian")
    #perform inverse fft on butterworth filtered image
    reconstructed_ishifted = np.fft.ifftshift(filtered_butter)
    reconstructed_ishifted_ifft = np.fft.ifft2(reconstructed_ishifted).real
    plotimg(reconstructed_ishifted_ifft,3,2,5,"Reconstructed image using Butterworth")
    
    plt.tight_layout()
    plt.show()
main()