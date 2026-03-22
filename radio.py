import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


from PIL import Image
image = Image.open('radiographie.jpg').convert('L') 
image_np = np.array(image)


sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])


grad_x = ndimage.convolve(image_np, sobel_x)
grad_y = ndimage.convolve(image_np, sobel_y)

# Magnitude du gradient 
edges = np.hypot(grad_x, grad_y)
edges = (edges / np.max(edges)) * 255  # Normalisation

# Affichage
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Image d\'origine')
plt.imshow(image_np, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Contours détectés')
plt.imshow(edges, cmap='gray')
plt.show()

