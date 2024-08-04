import cv2
import matplotlib.pyplot as plt
import numpy as np


def to_gray(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.jpg', gray_image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def show_gray_histogramm(image_path):
    gray_image = cv2.imread(image_path)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.figure()
    plt.title('Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()
    

def constrast_limit(image_path, min, max):
    image = cv2.imread(image_path)
    contrast_limited_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    cv2.imwrite('contrast_limited_image.jpg', contrast_limited_image)
    cv2.imshow('Contrast Limited Image', contrast_limited_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def convert_grayscale_range(image_path, min_val, max_val):
    # Step 1: Read the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image is read properly
    if image is None:
        raise ValueError("Image not found or unable to read the image file")
    
    # Step 2: Normalize the image to range 0-1
    normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Step 3: Scale the normalized values to the range [min_val, max_val]
    scaled_image = (normalized_image * (max_val - min_val)) + min_val
    
    # Step 4: Convert back to uint8
    limited_image = np.uint8(scaled_image)
    
    # Display the original and processed images
    cv2.imshow('Original Image', image)
    cv2.imshow('Limited Grayscale Image', limited_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the processed image (optional)
    cv2.imwrite('limited_grayscale_image.jpg', limited_image)
    
    
    
def read_rgba_histogram(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the image is read properly
    if image is None:
        raise ValueError("Image not found or unable to read the image file")
    
    # Check if the image has 4 channels (RGBA)
    if image.shape[2] != 3:
        raise ValueError("Image does not have 3 channels (RGBA)")

    # Step 2: Split the image into its respective channels
    blue, green, red = cv2.split(image)

    # Step 3: Calculate histograms for each channel
    hist_red = cv2.calcHist([red], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([green], [0], None, [256], [0, 256])
    hist_blue = cv2.calcHist([blue], [0], None, [256], [0, 256])


    # Step 4: Plot the histograms
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(hist_red, color='red')
    plt.title('Red Channel Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    
    plt.subplot(2, 2, 2)
    plt.plot(hist_green, color='green')
    plt.title('Green Channel Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    
    plt.subplot(2, 2, 3)
    plt.plot(hist_blue, color='blue')
    plt.title('Blue Channel Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    

    plt.tight_layout()
    plt.show()
    
    
def invert_gray_image(image_path):
    # Step 1: Read the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image is read properly
    if image is None:
        raise ValueError("Image not found or unable to read the image file")
    
    # Step 2: Invert the image by subtracting from 255
    inverted_image = 255 - image
    
    # Display the original and inverted images
    cv2.imshow('Original Image', image)
    cv2.imshow('Inverted Image', inverted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the inverted image (optional)
    cv2.imwrite('inverted_image.jpg', inverted_image)
    
    
    
def increase_brightness(image_path, x):
    # Step 1: Read the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image is read properly
    if image is None:
        raise ValueError("Image not found or unable to read the image file")
    
    # Step 2: Increase the brightness by adding x to each pixel
    brightened_image = cv2.add(image, np.array([x], dtype=np.uint8))
    
    # Display the original and brightened images
    cv2.imshow('Original Image', image)
    cv2.imshow('Brightened Image', brightened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the brightened image (optional)
    cv2.imwrite('brightened_image.jpg', brightened_image)
    
    
def decrease_brightness(image_path, x):
    # Step 1: Read the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image is read properly
    if image is None:
        raise ValueError("Image not found or unable to read the image file")
    
    # Step 2: Decrease the brightness by subtracting x from each pixel
    decreased_image = cv2.subtract(image, np.array([x], dtype=np.uint8))
    
    # Display the original and decreased brightness images
    cv2.imshow('Original Image', image)
    cv2.imshow('Decreased Brightness Image', decreased_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the decreased brightness image (optional)
    cv2.imwrite('decreased_brightness_image.jpg', decreased_image)
    
    
def gaussian_blur_3x3(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Define a 3x3 Gaussian kernel
    gaussian_kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=np.float32) / 16

    # Apply Gaussian blur using the kernel
    blurred_image = cv2.filter2D(image, -1, gaussian_kernel)
    cv2.imshow('Original Image', image)
    cv2.imshow('Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the decreased brightness image (optional)
    cv2.imwrite('3x3_blurred_image.jpg', blurred_image)

def average_blur(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Define a 3x3 Gaussian kernel
    average_kernel = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]], dtype=np.float32) / 9

    # Apply Gaussian blur using the kernel
    blurred_image = cv2.filter2D(image, -1, average_kernel)
    cv2.imshow('Original Image', image)
    cv2.imshow('Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the decreased brightness image (optional)
    cv2.imwrite('blurred_image.jpg', blurred_image)
    
    
def gaussian_blur_7x7(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Define a 7x7 Gaussian kernel
    gaussian_kernel = np.array([[1,   6,  15,     20,  15,  6,  1],
                                [6,  36, 90,      120, 90,  36,  6],
                                [15,  90, 225,    300, 225, 90,  15],
                                [20, 120, 300,    400, 300, 120, 20],
                                [15,  90, 225,    300, 225, 90,  15],
                                [6,  36, 90,      120, 90,  36,  6],
                                [1,   6,  15,     20,  15,  16,  1]], dtype=np.float32)

    # Normalize the kernel
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    # Apply Gaussian blur using the kernel
    blurred_image = cv2.filter2D(image, -1, gaussian_kernel)

    cv2.imshow('Original Image', image)
    cv2.imshow('Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the blurred image (optional)
    cv2.imwrite('7x7_blurred_image.jpg', blurred_image)


to_gray('1311862.jpeg')
# show_gray_histogramm('gray_image.jpg')
# convert_grayscale_range('gray_image.jpg', 100, 150)
# show_gray_histogramm('limited_grayscale_image.jpg')
# show_gray_histogramm('image.jpeg')
# read_rgba_histogram('image.jpeg')
# invert_gray_image('gray_image.jpg')
# show_gray_histogramm('gray_image.jpg')
# show_gray_histogramm('inverted_image.jpg')
# show_gray_histogramm('brightened_image.jpg')
# show_gray_histogramm('decreased_brightness_image.jpg')
# increase_brightness('gray_image.jpg', 50)
# decrease_brightness('gray_image.jpg', 50)
# gaussian_blur_3x3('image.jpeg')
# average_blur('image.jpeg') 
# gaussian_blur_7x7('image.jpeg') 