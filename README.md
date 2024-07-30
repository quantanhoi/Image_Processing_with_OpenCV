# Image Processing Functions

This repository contains various image processing functions implemented in `script.py` using OpenCV and Matplotlib.

## Functions

### `to_gray(image_path)`

Converts a color image to grayscale.

- **Parameters:**
  - `image_path` (str): Path to the input image.

### `show_gray_histogramm(image_path)`

Displays the histogram of a grayscale image.

- **Parameters:**
  - `image_path` (str): Path to the input image.

### `constrast_limit(image_path, min, max)`

Applies contrast limiting to an image.

- **Parameters:**
  - `image_path` (str): Path to the input image.
  - `min` (int): Minimum contrast value.
  - `max` (int): Maximum contrast value.

### `convert_grayscale_range(image_path, min_val, max_val)`

Converts the grayscale range of an image.

- **Parameters:**
  - `image_path` (str): Path to the input image.
  - `min_val` (int): Minimum grayscale value.
  - `max_val` (int): Maximum grayscale value.

### `read_rgba_histogram(image_path)`

Reads and displays the histogram of an RGBA image.

- **Parameters:**
  - `image_path` (str): Path to the input image.

### `invert_gray_image(image_path)`

Inverts a grayscale image.

- **Parameters:**
  - `image_path` (str): Path to the input image.

### `increase_brightness(image_path, x)`

Increases the brightness of a grayscale image.

- **Parameters:**
  - `image_path` (str): Path to the input image.
  - `x` (int): Value to increase the brightness by.

### `decrease_brightness(image_path, x)`

Decreases the brightness of a grayscale image.

- **Parameters:**
  - `image_path` (str): Path to the input image.
  - `x` (int): Value to decrease the brightness by.

### `gaussian_blur_3x3(image_path)`

Applies a 3x3 Gaussian blur to an image.

- **Parameters:**
  - `image_path` (str): Path to the input image.

### `average_blur(image_path)`

Applies an average blur to an image using a 3x3 kernel.

- **Parameters:**
  - `image_path` (str): Path to the input image.

### `gaussian_blur_7x7(image_path)`

Applies a 7x7 Gaussian blur to an image.

- **Parameters:**
  - `image_path` (str): Path to the input image.