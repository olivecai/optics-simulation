import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import time

def load_image(filepath: str) -> np.array: 
    try:
        img=Image.open(filepath)
        print("Image loaded succesfully")
        return np.moveaxis(np.array(img), 0, 1)
    except Exception as e:
        print("Error loading image: {e}")
        return None


def view_image(image: np.array, title: str = None) -> None: 
    print(image.shape)
    plt.imshow(image)
    if title is not None: 
        plt.title(title)
    plt.show()

def create_gaussian_kernel(size:int, stddev:float) -> np.array: 
    '''
    This function generates a 1D Gaussian distribution using numpy "linspace".
    '''
    # Generate `size` points between -stddev and stddev for a symmetric Gaussian distribution
    x = np.linspace(-size//2, size//2, size)
    # Compute the 1D Gaussian distribution using the PDF of the normal distribution
    gaussian_1d = scipy.stats.norm.pdf(x, loc=0, scale=stddev)
    # Normalize to make the sum equal to 1
    gaussian_1d /= gaussian_1d.sum()
    # Create the 2D Gaussian kernel by taking the outer product of the 1D Gaussian with itself
    gaussian_kernel_2d = np.outer(gaussian_1d, gaussian_1d)
    return gaussian_kernel_2d

    
def convolve2d(x: np.array, h: np.array) -> np.array: 
    """
    x: a 2d or 3d np.array representing your image 
    h: a 2d np.array representing your kernel 
    """ 
    start = time.time()  
    print("Computing convolution... may take a few minutes...")
    print("At this point, just quit the program... it took 10 minutes on my machine, its probably going to take just as long on yours...")
    print("Move on to the FFT Convolution")
    if x.ndim == 3:
        channels = [scipy.signal.convolve2d(x[:, :, i], h, mode="same", boundary="wrap") for i in range(x.shape[-1])]
        y = np.stack(channels, axis=-1)
    else: 
        assert x.ndim == 2
        y =  scipy.signal.convolve2d(x, h, mode="same", boundary="wrap")

    stop = time.time() 
    print(f"Convolution took {stop - start} seconds")
    return y.astype(np.int32)


def fft_conv2d(x: np.array, h:np.array) -> np.array:
    """
    The trick of computing convolutions quickly: Fourier transform your signal and kernel
    to the frequency domain, and perform a multiplication instead!


    x: np.array[H, W, C]: A 3d tensor representing an image
    h: np.array[K, K]: a kernel you want to convolve with the image  
    """

    start1 = time.time() 

    # For multiplication, we want our signal and kernel to have the same spatial dimensions.
    # For the kernel, simply zero-pad it until it reaches the desired shape
    h_padded = np.zeros_like(x[:, :, 0]).astype(h.dtype) 
    startY = (x.shape[0] - h.shape[0])//2
    startX = (x.shape[1] - h.shape[1])//2
    h_padded[startY:startY + h.shape[0], startX:startX + h.shape[1]] += h

    # 2D FFT. We first swap the axis of X, so that it goes from [H, W, C] to [C, H, W].
    # This is in accordance to the documentation for np.fft.fft2
    X = np.fft.fft2(np.moveaxis(x, -1, 0))
    H = np.fft.fft2(h_padded)

    stop1 = time.time()

    # visualizations are important!
    view_image(np.moveaxis(np.real(X), 0, -1), "FT of image")
    view_image(np.real(H), "FT of kernel")


    start2 = time.time() 

    # Convolution theorem: FT{response} = FT{signal} x TRANSFER FUNCTION
    Y = X * H[None, :, :] # This is still in the form [C, H, W]

    # Inverse transform
    y = np.fft.ifft2(Y)

    # This part will shift all the frequencies back to the right place, since discrete fourier transforms 
    # are not centered around the origin. Uncommenting this equation will result in shifted colours in the 
    # resulting image
    y = np.stack(
        [np.fft.fftshift(y[i]) for i in range(y.shape[0])], 
        axis=0)
    y = np.moveaxis(y, 0, -1)

    stop2  = time.time() 

    print(f"FFT Convolution took {stop2 - start2 + stop1 - start1} seconds")
    return np.real(y).astype(np.int32)



def olivia_fft_conv2d(x: np.array, h:np.array) -> np.array:
    """
    Now its your turn! Be careful of the axes, as we will move the channel axis around and back... 

    The trick of computing convolutions quickly: Fourier transform your signal and kernel
    to the frequency domain, and perform a multiplication instead!

    x: np.array[H, W, C]: A 3d tensor representing an image
    h: np.array[K, K]: a kernel you want to convolve with the image  
    """


    # For multiplication, we want our signal and kernel to have the same spatial dimensions.
    # For the kernel, simply zero-pad it until it reaches the desired shape


    # 2D FFT. We first swap the axis of X, so that it goes from [H, W, C] to [C, H, W].
    # This is in accordance to the documentation for np.fft.fft2


    # visualizations are important!



    # Convolution theorem: FT{response} = FT{signal} x TRANSFER FUNCTION

    # Inverse transform

    # This part will shift all the frequencies back to the right place, since discrete fourier transforms 
    # are not centered around the origin. Uncommenting this equation will result in shifted colours in the 
    # resulting image

    pass





def main():
    filepath = "temp_test_photo.JPG"
    img=load_image(filepath)
    view_image(img, "original image")

    print("gaussian kernel visualization")
    h_gaussian = create_gaussian_kernel(size=100, stddev=30)
    view_image(h_gaussian, "gaussian kernel")

    # Comment this out if you don't want to wait 10 minutes....
    result = convolve2d(img, h_gaussian)
    view_image(result, "convolution of kernel with image")

    result = fft_conv2d(img, h_gaussian)
    view_image(result)



if __name__ == "__main__": 
    main()