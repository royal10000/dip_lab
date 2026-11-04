
import cv2
import numpy as np
from show import showImg

img = cv2.imread("images/cat.jpg")

if img is None:
    raise FileNotFoundError("Image not found")

f = img.astype(np.float32)

# Contrast enhancement
contrast = 2
contrast_float = np.clip(f * contrast, 0, 255)
contrast_img = contrast_float.astype(np.uint8)

noise_sigma = [12.0, 25.0]
number_of_images = [1, 10,20]
seed = 7

rng = np.random.default_rng(seed)

images = [contrast_img]
titles = ["Contrast reference"]

for sigma in noise_sigma:
    noisy_images = []

    for i in range(max(number_of_images)):
        noise = rng.normal(0, sigma, contrast_float.shape)
        noisy = np.clip(contrast_float + noise, 0, 255)
        noisy_images.append(noisy)

    for count in number_of_images:
        stack = np.stack(noisy_images[:count], axis=0)

        average_float = np.mean(stack, axis=0)
        average = np.clip(average_float, 0, 255).astype(np.uint8)

        # MSE against contrast-enhanced reference
        mse = np.mean((average_float - contrast_float) ** 2)

        print(f"Sigma={sigma}, Images={count}, MSE={mse:.2f}")

        if count == 1:
            images.append(average)
            titles.append(f"Noise {sigma}, one image")
        else:
            images.append(average)
            titles.append(f"Noise {sigma}, average of {count}")

showImg(images, titles)

#Prediction before running
"""
When the noise level stays at (sigma=12):
- (N=1): highest expected noise and MSE.
- (N=10): lower expected noise and MSE.
- (N=20): lowest expected noise and MSE.
When you increase the noise level to (sigma=25) and use (N=20), averaging should still reduce the noise, but the result may remain noisier
than the (sigma=12, N=20) result.
"""

# MSE Output 
"""
Sigma=12.0, Images=1, MSE=123.70
Sigma=12.0, Images=10, MSE=17.52
Sigma=12.0, Images=20, MSE=11.64
Sigma=25.0, Images=1, MSE=514.26
Sigma=25.0, Images=10, MSE=75.52
Sigma=25.0, Images=20, MSE=51.12
"""

#Explaination in my own words
"""
Averaging simulated aligned images reduces noise because each image contains different random noise. When we average many images, the random 
noise tends to cancel out, making the result closer to the clean reference image and reducing MSE. Increasing the noise level to 25 makes the
noise stronger, although averaging 20 images still reduces it. However, repeatedly averaging identical copies of one photograph does not
reduce its existing noise because every copy contains the same pixel values and noise.
"""