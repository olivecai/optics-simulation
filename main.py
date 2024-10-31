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
    print("Shape of the image being displayed:",image.shape)
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
    ###### The discrete Fourier transform is a circular convolution; thus, values wrap around and may create artifacts on the edge of our image unless we zero pad it.
    x_shape= np.shape(x)
    h_shape=np.shape(h)
    ''' 
    Below was my initial (naive sad onerous) approach.   
    #while x-dimension of arrays h and x are different, alternate padding on right and left of h. Check in between R and L if sizes are equal.
    while x_shape[0] != h_shape[0]:
        h= np.pad(h, ((1,0),(0,0)), 'constant', constant_values=0) #pad the left of the array
        h_shape=np.shape(h)
        if x_shape[0] == h_shape[0]:
            break
        h=np.pad(h, ((0,1),(0,0)), 'constant', constant_values=0) #pad the right of the array
        h_shape=np.shape(h)
    #same for y-dimension
    while x_shape[1] != h_shape[1]:
        h=np.pad(h, ((0,0),(1,0)), 'constant', constant_values=0) #pad the left of the array
        h_shape=np.shape(h)
        if x_shape[1] == h_shape[1]:
            break
        h=np.pad(h, ((0,0),(0,1)), 'constant', constant_values=0) #pad the right of the array
        h_shape=np.shape(h) 
        '''
    #Separately calculate how many rows/cols are needed to pad onto h 
    pad_x = x_shape[0] - h_shape[0]
    pad_y = x_shape[1] - h_shape[1]
    pad_right = pad_x//2
    pad_left = pad_x - pad_right
    pad_up = pad_y//2
    pad_down = pad_y - pad_up

    h_padded=np.pad(h, ((pad_right, pad_left), (pad_up, pad_down)), 'constant', constant_values=0)
    print("this was h originally:", h)
    print("this is h after padding:", h_padded)
    print("This is the shape of x", np.shape(x))
    print("This is the shape of h", np.shape(h))

    # 2D FFT. We first swap the axis of X, so that it goes from [H, W, C] to [C, H, W].
    # This is in accordance to the documentation for np.fft.fft2
    ###### We swap the axis because np.fft.fft2 automatically performs the FT on the last two dimensions.
    x_swapped_axis = np.moveaxis(x, -1, 0)
    X = np.fft.fft2(x_swapped_axis)  # dimensions CxHxW, because Channels unimportant rn
    H = np.fft.fft2(h_padded)  # h_padded is already dimensions HxW
    print("This is the shape of X", np.shape(X))
    print("This is the shape of H", np.shape(H))

    # visualizations are important!
    ###### When viewing X and H, be cognizant of two things: 
    ############ 1. We only want to view the real portion (because thats the amplitude of our frequency). 
    ############ 2. We need to swap X axis again to H,W,C because it's common practice for Channels (color) to be the third dimension.
    img_X = np.moveaxis(np.real(X), 0, -1)
    img_H = np.real(H)
    view_image(img_X, "Olivia's FT of image!")
    view_image(img_H, "Olivia's FT of kernel!")

    # Convolution theorem: FT{response} = FT{signal} x TRANSFER FUNCTION
    ####### We want H to be the same dimensions as X so that we can do element-wise multiplication. 
    ####### Note that this is NOT matrix multiplication. Numpy has different syntax for that. 
    print("This is the shape of X", np.shape(X))
    print("This is the shape of H", np.shape(H))
    
    Y = X * H[None,:,:]   #

    # Inverse transform!
    y= np.fft.fft2(Y)

    # This part will shift all the frequencies back to the right place, since discrete fourier transforms are not centered around the origin. Uncommenting this equation will result in shifted colours in the resulting image
    ###### Remember that it is still C, H, W. We want to shift the frequencies back to normal so that our colors appear normally.
    ###### Remember that the first channel is for colors? Iterate through the range of the colors. 
    ## QUESTION: in this case, its 3, right? (RGB?)
    color_channel_dimension = y.shape[0]
    arrays=[np.fft.fftshift(y[i]) for i in range(color_channel_dimension)]
    y=np.stack(arrays, axis=0)  #axis=0 refers to the first dimension. i elements in y.shape[0] will lead to resulting y array having dimension (i,H,W)
    # Dont forget that y is still arranged as C H W! Let's put that color channel back where it belongs.
    y=np.moveaxis(y, 0,-1)

    ## Yay! Return your array, the convolved image! Remember: for viewing the image, we only care about real elements.
    ## Note: integers represent colors, etc. If we don't include the conversion to int32, we see a blank white picture!
    return np.real(y).astype(np.int32)


def main():

    filepath = "temp_test_photo.JPG"
    img=load_image(filepath)
    view_image(img, "original image")

    print("gaussian kernel visualization")
    h_gaussian = create_gaussian_kernel(size=100, stddev=30)
    view_image(h_gaussian, "gaussian kernel")

    # Comment this out if you don't want to wait 10 minutes....
    #result = convolve2d(img, h_gaussian)
    #view_image(result, "convolution of kernel with image")

    result = fft_conv2d(img, h_gaussian)
    view_image(result)

    ################ Olivia Testing Zone!
    result= olivia_fft_conv2d(img, h_gaussian)
    view_image(result, "Olivia's final image")




if __name__ == "__main__": 
    main()