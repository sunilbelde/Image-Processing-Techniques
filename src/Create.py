#Importing the required packages
import numpy as np
import cv2
import math
import os 

#Creating output folder if does not exist
if not os.path.isdir('output'):
  os.mkdir('output')

def addNoise(image,magnitude):
  '''This function adds noise to the image'''
  temp=[]
  for x in np.nditer(image):
    random=np.random.rand()
    if magnitude>random: # If magnitude is greater than random numeber then add noise to that pixel
      x=np.random.randint(255)
    temp.append(x)
  temp=np.array(temp,dtype=np.uint8).reshape(image.shape[0],image.shape[1],image.shape[2])  
  return temp
img=cv2.imread('input/princeton_small.jpg')


#Genereating some output noise images
if not os.path.isfile('output/princeton_small_noise_img_0.3.jpg'):
  noise_img=addNoise(img,0.3)
  cv2.imwrite('output/princeton_small_noise_img_0.3.jpg', noise_img)

if not os.path.isfile('output/princeton_small_noise_img_0.5.jpg'):
  noise_img=addNoise(img,0.5)
  cv2.imwrite('output/princeton_small_noise_img_0.5.jpg', noise_img)

if not os.path.isfile('output/princeton_small_noise_img_1.0.jpg'):
  noise_img=addNoise(img,1)
  cv2.imwrite('output/princeton_small_noise_img_1.0.jpg', noise_img)


def Brighten(image,magnitude):
  '''This funtion Brightesns the given input images'''
  temp=[]
  for x in np.nditer(image):
    x=x*magnitude
    if x>255:
      x=255
    temp.append(x)
  temp=np.array(temp,dtype=np.uint8).reshape(image.shape[0],image.shape[1],image.shape[2])  
  return temp

img=cv2.imread('input/princeton_small.jpg')

#Genereating some output brightened images
if not os.path.isfile('output/princeton_small_brighten_img_0.0.jpg'):
  brighten_img=Brighten(img,0)
  cv2.imwrite('output/princeton_small_brighten_img_0.0.jpg', brighten_img)

if not os.path.isfile('output/princeton_small_brighten_img_0.5.jpg'):
  brighten_img=Brighten(img,0.5)
  cv2.imwrite('output/princeton_small_brighten_img_0.5.jpg', brighten_img)
  
if not os.path.isfile('output/princeton_small_brighten_img_2.0.jpg'):
  brighten_img=Brighten(img,2)
  cv2.imwrite('output/princeton_small_brighten_img_2.0.jpg', brighten_img)


#https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
def ChangeContrast(image,magnitude):
  '''This function Generates contrast images'''
  temp=[]
  magnitude=magnitude*(255)-255
  f=(259*(255+magnitude))/(255*(259-magnitude))
  for x in np.nditer(image):
    x=f*(x-128)+128
    temp.append(x)
  temp=np.array(temp,dtype=np.uint8).reshape(image.shape[0],image.shape[1],image.shape[2])  
  return temp
  
img=cv2.imread('input/c.jpg')

if not os.path.isfile('output/c_contrast_img_-0.5.jpg'):
  contrast_img=ChangeContrast(img,-0.5)
  cv2.imwrite('output/c_contrast_img_-0.5.jpg', contrast_img)

if not os.path.isfile('output/c_contrast_img_0.0.jpg'):
  contrast_img=ChangeContrast(img,0)
  cv2.imwrite('output/c_contrast_img_0.0.jpg', contrast_img)

if not os.path.isfile('output/c_contrast_img_0.5.jpg'):
  contrast_img=ChangeContrast(img,0.5)
  cv2.imwrite('output/c_contrast_img_0.5.jpg', contrast_img)

if not os.path.isfile('output/c_contrast_img_2.0.jpg'):
  contrast_img=ChangeContrast(img,2)
  cv2.imwrite('output/c_contrast_img_2.0.jpg', contrast_img)


#https://www.imageeprocessing.com/2014/04/gaussian-filter-without-using-matlab.html
def get_gaussian_filter(sigma):
  '''This function gets the Gaussian filter for corresponding sigma'''
  filter_width=int(math.ceil(3*sigma)*2+1)
  x=filter_width//2
  gaussian_filter=np.zeros((filter_width,filter_width))
  for i in range(-x,x+1):
    for j in range(-x,x+1):
      numerator=math.exp(-1*(i**2+j**2)/(2*(sigma**2)))
      denominator=2*math.pi*(sigma**2)
      gaussian_filter[i][j]=numerator/denominator
  return gaussian_filter
  
  
  
def Blur(image,magnitude):
  '''This function blurs the images'''
  sigma=magnitude
  gaussian_filter=get_gaussian_filter(sigma)
  blurred_image= np.zeros((image.shape[0]-gaussian_filter.shape[0], image.shape[1]-gaussian_filter.shape[0],image.shape[2]))
  for x in range(image.shape[0]-gaussian_filter.shape[0]):
    for y in range(image.shape[1]-gaussian_filter.shape[1]):
      for z in range(image.shape[2]):
        blurred_image[x,y,z]=(gaussian_filter * image[x: x+gaussian_filter.shape[0], y: y+gaussian_filter.shape[1],z]).sum() # Performing COnvlution operation
  blurred_image=np.array(blurred_image,dtype=np.uint8)
  return blurred_image


img=cv2.imread('input/princeton_small.jpg')

if not os.path.isfile('output/princeton_small_blur_img_0.5.jpg'):
  blur_img=Blur(img,0.5)
  cv2.imwrite('output/princeton_small_blur_img_0.5.jpg', blur_img)

if not os.path.isfile('output/princeton_small_blur_img_2.0.jpg'):
  blur_img=Blur(img,2)
  cv2.imwrite('output/princeton_small_blur_img_2.0.jpg', blur_img)
  
if not os.path.isfile('output/princeton_small_blur_img_8.0.jpg'):
  blur_img=Blur(img,8)
  cv2.imwrite('output/princeton_small_blur_img_8.0.jpg', blur_img)


def Sharpen(image):
  '''This Function Sharpens the Images'''
  #sigma=magnitude
  sharpen_filter=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
  image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2,image.shape[2])) # Padding the input image
  image_padded[1:-1, 1:-1,:] = image
  sharpened_image= np.zeros((image.shape[0], image.shape[1],image.shape[2]))
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      for z in range(image.shape[2]):
        sharpened_image[x,y,z]=(sharpen_filter * image_padded[x: x+3, y:y+3,z]).sum() # Performing COnvlution operation
  sharpened_image=np.array(sharpened_image,dtype=np.uint8)
  return sharpened_image

img=cv2.imread('input/princeton_small.jpg')

if not os.path.isfile('output/princeton_small_sharpen.jpg'):
  sharpen_img=Sharpen(img)
  cv2.imwrite('output/princeton_small_sharpen.jpg', sharpen_img)


def EdgeDetect(image):
  '''This Function is used for edge detection'''
  edgeDetect_filter=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) 
  image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2,image.shape[2])) # Padding the input image
  image_padded[1:-1, 1:-1,:] = image
  egde_detector_image= np.zeros((image.shape[0], image.shape[1],image.shape[2]))
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      for z in range(image.shape[2]):
        egde_detector_image[x,y,z]=abs((edgeDetect_filter * image_padded[x: x+3, y:y+3,z]).sum()) # Performing COnvlution operation
  egde_detector_image=np.array(egde_detector_image,dtype=np.uint8)
  return egde_detector_image

img=cv2.imread('input/cube.jpg')

if not os.path.isfile('output/cube_edge_detector.jpg'):
  edge_detector_img=EdgeDetect(img)
  cv2.imwrite('output/cube_edge_detector.jpg', edge_detector_img)


def composite(image,top,mask):
  '''Tbis function performs the composite operation on given base,top and mask images'''
  output=np.zeros(image.shape)
  output=np.array(output,dtype=np.uint8)
  for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):
      if mask[x,y,:].sum()<=255:
        output[x,y,:]=image[x,y,:]
      else:
        output[x,y,:]=top[x,y,:]
  return output

image=cv2.imread('input/comp_background.jpg')
top=cv2.imread('input/comp_foreground.jpg')
mask=cv2.imread('input/comp_mask.jpg')

if not os.path.isfile('output/composite_img.jpg'):
  composite_img=composite(image,top,mask)
  cv2.imwrite('output/composite_img.jpg', composite_img)

def Upscaling(image,scale):
  image_upscaled = np.zeros((image.shape[0] * scale, image.shape[1] * scale,image.shape[2]))
  i,j=0,0
  for x in range(image.shape[0]):
    j=0
    for y in range(image.shape[1]):
      for s in range(scale):
        image_upscaled[i+s,j+s,:]=image[x,y,:]
      j+=scale
    i+=scale
  image_upscaled=np.array(image_upscaled,dtype=np.uint8)
  return image_upscaled

img=cv2.imread('input/princeton_small.jpg')

if not os.path.isfile('output/princeton_small_upscale_2.jpg'):
  edge_detector_img=Upscaling(img,2)
  cv2.imwrite('output/princeton_small_upscale_2.jpg', edge_detector_img)
  

if not os.path.isfile('output/princeton_small_upscale_3.jpg'):
  edge_detector_img=Upscaling(img,3)
  cv2.imwrite('output/princeton_small_upscale_3.jpg', edge_detector_img)


print("All filters applied..")