import numpy as np
from PIL import Image
import matplotlib as plt
import scipy

def load_image(filepath: str) -> np.array: 
    try:
        img=Image.open(filepath)
        print("Image loaded succesfully")
        return np.array(img)
    except Exception as e:
        print("Error loading image: {e}")
        return None


def view_image(image: np.array) -> None: 
    plt.imshow(image)
    plt.show()

def create_gaussian_kernel(size:int, stddev:float) -> np.array: 
    '''
    This function generates a 1D Gaussian distribution using numpy "linspace".
    '''
    # Generate `size` points between -stddev and stddev for a symmetric Gaussian distribution
    x = np.linspace(-stddev, stddev, size)
    
    # Compute the 1D Gaussian distribution using the PDF of the normal distribution
    gaussian_1d = scipy.stats.norm.pdf(x, loc=0, scale=stddev)
    
    # Normalize to make the sum equal to 1
    gaussian_1d /= gaussian_1d.sum()
    
    # Create the 2D Gaussian kernel by taking the outer product of the 1D Gaussian with itself
    gaussian_kernel_2d = np.outer(gaussian_1d, gaussian_1d)
    
    return gaussian_kernel_2d

def main():
    filepath = NotImplemented ################################ ADD FILE PATH
    img=load_image(filepath)
    view_image(img)