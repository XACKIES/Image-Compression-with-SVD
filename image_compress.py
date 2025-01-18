import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color


image = io.imread('Maku.jpg')  
gray_image = color.rgb2gray(image)    
A = np.array(gray_image)              

# Perform SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

# Compress the image 
k = 60
U_k = U[:, :k]        
S_k = np.diag(S[:k])  
VT_k = VT[:k, :]     

# Compressed Image 
compressed_image = np.dot(U_k, np.dot(S_k, VT_k))

# Calculate Reconstruction Error
reconstruction_error = np.linalg.norm(A - compressed_image) / np.linalg.norm(A)
print(f"Reconstruction Error: {reconstruction_error:.4%}")


plt.figure(figsize=(18, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(A, cmap='gray')
plt.axis('off')

# Compressed Image 
plt.subplot(1, 2, 2)
plt.title(f"Compressed Image (k={k})")
plt.imshow(compressed_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

