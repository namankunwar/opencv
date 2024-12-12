import cv2 as cv

img= cv.imread("photo/building.jpg")

def rescale(image, scale):
        
        """
    Rescales the input image by the given scale factor.
    :param img: Input image
    :param scale: Scale factor (e.g., 0.5 for half size, 2.0 for double size)
    :return: Rescaled image
    """
     
        width = int(img.shape[1]* scale) # 1 For width
        height = int(img.shape[0]* scale) # 0 for height
        dimensions = (width, height)
        return cv.resize(img, dimensions,interpolation= cv.INTER_AREA)

rescale_image= rescale(img, 0.69)

resize_image = cv.resize(img, (800,700), interpolation= cv.INTER_LINEAR)

cv.imshow("orginal image", img)
cv.imshow("rescale image", rescale_image)
cv.imshow("resize image with 800*700")