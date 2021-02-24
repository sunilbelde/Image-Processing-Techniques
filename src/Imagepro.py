"""
This python has following functions implemenented: 

All the generated output files are saved in output folder

->Brighten(filename,magnitude) : This function takes image filename, magnitude of filter to be applied.

->addNoise(filename,magnitude) : This function takes image filename, magnitude of filter to be applied.

->ChangeContrast(filename,magnitude) : This function takes image filename, magnitude of filter to be applied.

->Blur(filename,magnitude) : This function takes image filename, magnitude of filter to be applied.

->Sharpen(filename) : This function takes image filename as argument and generates sharpened image.

->EdgeDetect(filename) : This function takes image filename as argument and generates edge detector image.

->composite(image,top,mask): This function takes three images files (base image,top image and mask image) and performs compute operation.
"""
import sys
import math
import numpy as np 
import cv2
import os

def Brighten(filename,magnitude):
  image=cv2.imread('input/'+filename)
  magnitude=float(magnitude)
  temp=[]
  for x in np.nditer(image):
    x=x*magnitude
    if x>255:
      x=255
    temp.append(x)
  temp=np.array(temp,dtype=np.uint8).reshape(image.shape[0],image.shape[1],image.shape[2])  
  output_file='output/'+filename.split('.')[0]+'_brighten_img_'+str(magnitude)+'.jpg'
  cv2.imwrite(output_file, temp)
  print('Saved image to outout folder as '+output_file)
  

def addNoise(filename,magnitude):
  image=cv2.imread('input/'+filename)
  magnitude=float(magnitude)
  temp=[]
  for x in np.nditer(image):
    random=np.random.rand()
    if magnitude>random:
      x=np.random.randint(255)
    temp.append(x)
  temp=np.array(temp,dtype=np.uint8).reshape(image.shape[0],image.shape[1],image.shape[2])  
  
  output_file='output/'+filename.split('.')[0]+'_noise_img_'+str(magnitude)+'.jpg'
  cv2.imwrite(output_file, temp)
  print('Saved image to outout folder as '+output_file)

def ChangeContrast(filename,magnitude):
  image=cv2.imread('input/'+filename)
  mag=float(magnitude)
  magnitude=float(magnitude)
  temp=[]
  magnitude=magnitude*(255)-255
  f=(259*(255+magnitude))/(255*(259-magnitude))
  for x in np.nditer(image):
    x=f*(x-128)+128
    temp.append(x)
  temp=np.array(temp,dtype=np.uint8).reshape(image.shape[0],image.shape[1],image.shape[2])  
  
  output_file='output/'+filename.split('.')[0]+'_contrast_img_'+str(mag)+'.jpg'
  cv2.imwrite(output_file, temp)
  print('Saved image to outout folder as '+output_file)


def get_gaussian_filter(sigma):
  filter_width=int(math.ceil(3*sigma)*2+1)
  x=filter_width//2
  gaussian_filter=np.zeros((filter_width,filter_width))
  for i in range(-x,x+1):
    for j in range(-x,x+1):
      numerator=math.exp(-1*(i**2+j**2)/(2*(sigma**2)))
      denominator=2*math.pi*(sigma**2)
      gaussian_filter[i][j]=numerator/denominator
  return gaussian_filter
  
  
  
def Blur(filename,magnitude):
  image=cv2.imread('input/'+filename)
  magnitude=float(magnitude)
  sigma=magnitude
  gaussian_filter=get_gaussian_filter(sigma)
  blurred_image= np.zeros((image.shape[0]-gaussian_filter.shape[0], image.shape[1]-gaussian_filter.shape[0],image.shape[2]))
  for x in range(image.shape[0]-gaussian_filter.shape[0]):
    for y in range(image.shape[1]-gaussian_filter.shape[1]):
      for z in range(image.shape[2]):
        blurred_image[x,y,z]=(gaussian_filter * image[x: x+gaussian_filter.shape[0], y: y+gaussian_filter.shape[1],z]).sum()
  blurred_image=np.array(blurred_image,dtype=np.uint8)
  
  output_file='output/'+filename.split('.')[0]+'_blur_img_'+str(magnitude)+'.jpg'
  cv2.imwrite(output_file, blurred_image)
  print('Saved image to outout folder as '+output_file)
  
def Sharpen(filename):
  image=cv2.imread('input/'+filename)
  sharpen_filter=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
  image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2,image.shape[2]))
  image_padded[1:-1, 1:-1,:] = image
  sharpened_image= np.zeros((image.shape[0], image.shape[1],image.shape[2]))
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      for z in range(image.shape[2]):
        sharpened_image[x,y,z]=(sharpen_filter * image_padded[x: x+3, y:y+3,z]).sum()
  sharpened_image=np.array(sharpened_image,dtype=np.uint8)
  
  output_file='output/'+filename.split('.')[0]+'_sharpen.jpg'
  cv2.imwrite(output_file, sharpened_image)
  print('Saved image to outout folder as '+output_file)

def EdgeDetect(filename):
  image=cv2.imread('input/'+filename)
  edgeDetect_filter=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) 
  image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2,image.shape[2]))
  image_padded[1:-1, 1:-1,:] = image
  egde_detector_image= np.zeros((image.shape[0], image.shape[1],image.shape[2]))
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      for z in range(image.shape[2]):
        egde_detector_image[x,y,z]=abs((edgeDetect_filter * image_padded[x: x+3, y:y+3,z]).sum())
  egde_detector_image=np.array(egde_detector_image,dtype=np.uint8)
  
  output_file='output/'+filename.split('.')[0]+'_edge_detector.jpg'
  cv2.imwrite(output_file, egde_detector_image)
  print('Saved image to outout folder as '+output_file)


def Upscaling(filename,scale):
  image=cv2.imread('input/'+filename)
  scale=int(scale)
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
  
  output_file='output/'+filename.split('.')[0]+'_upscale_'+str(scale)+'.jpg'
  cv2.imwrite(output_file, image_upscaled)
  print('Saved image to outout folder as '+output_file)

def composite(image,top,mask):
  image=cv2.imread('input/'+image)
  top=cv2.imread('input/'+top)
  mask=cv2.imread('input/'+mask)
  output=np.zeros(image.shape)
  output=np.array(output,dtype=np.uint8)
  for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):
      if mask[x,y,:].sum()<=255:
        output[x,y,:]=image[x,y,:]
      else:
        output[x,y,:]=top[x,y,:]
  output_file='output/composite_img.jpg'
  cv2.imwrite(output_file, output)
  print('Saved image to outout folder as '+output_file)

if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    if not os.path.isdir('output'):
      os.mkdir('output')
    if len(sys.argv)==2 and sys.argv[1]=='--help':
      print(__doc__)
    else:
      globals()[args[1]](*args[2:])