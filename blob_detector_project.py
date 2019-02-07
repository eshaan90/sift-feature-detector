#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:49:57 2018

@author: MyReservoir
"""


import cv2
import sys
import numpy as np
import time

def read_images(filelocation):
    image = cv2.imread(filelocation) 
    return image


def output_image_size(img,f,p):
    size=img.shape
    v_dim=size[0]+2*p-f+1
    h_dim=size[1]+2*p-f+1
    output_img=np.zeros((v_dim,h_dim), dtype=float)
    return output_img

def convolve(image,kernel):
    
    f=kernel.shape[0]
    iH, iW = np.shape(image)
    p=int((f-1)/2)
    padded_img=np.pad(image,p,'constant')
    output_img=output_image_size(image,f,p)
    
    for y in range(p, iH + p):
            for x in range(p, iW + p):
                roi = padded_img[y - p:y + p + 1, x - p:x + p + 1]
                k = (roi * kernel).sum()
                output_img[y - p, x - p] = k
                
    return output_img


def laplacian_of_gaussian_filter(sigma):
    kernel_size = np.round(6*sigma)
    if kernel_size % 2 == 0:
        kernel_size+=1
    half_size=np.floor(kernel_size/2)
    x, y = np.meshgrid(np.arange(-half_size, half_size+1), np.arange(-half_size, half_size+1))
    
    exp_term=np.exp(-(x**2+y**2) / (2*sigma**2))
    exp_term[exp_term < sys.float_info.epsilon * exp_term.max()] = 0
    if exp_term.sum() != 0:
        exp_term = exp_term/exp_term.sum() 
    else: 
        exp_term
    kernel = -((x**2 + y**2 - (2*sigma**2)) / sigma**2) * exp_term 
    kernel=kernel-kernel.mean()
    return kernel

def max_supression(scale_space, sigma, threshold_factor, level):
    max_scale_space = np.copy(scale_space)
    mask = [0]*(level)
    index = [(1, 0), (-1, 0), (0, 1), (0, -1), 
             (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for i in range(0,level):
        mask[i]=int(np.ceil(np.sqrt(2)*sigma[i]))
    size = np.shape(scale_space[:,:,0])
    
    def check(l):
        return all(scale_space[i + dx, j + dy, l] < scale_space[i, j, k] 
                   for dx, dy in index 
                   if  0<= i + dx < size[0] and 0<= j + dy <size[1])
    
    blob_location =[]
    for k in range(0,level):
        max_scale_space[:mask[k],:mask[k],k] = 0
        max_scale_space[-mask[k]:,-mask[k]:,k] = 0
        for i in range(mask[k]+1,size[0]-mask[k]-1):
            for j in range(mask[k]+1,size[1]-mask[k]-1):
                if scale_space[i, j, k] < threshold_factor:
                    continue
                c_max = check(k)
                l_max = u_max = True
                if k - 1 >= 0:
                    l_max = check(k - 1) and \
                    scale_space[i, j, k - 1] < scale_space[i, j, k]
                if k + 1 < level:
                    u_max = check(k + 1) and \
                    scale_space[i, j, k + 1] < scale_space[i, j, k]
                if c_max and l_max and u_max:
                    max_scale_space[i, j, k] = 1
                    blob_location.append((i,j,k))   
    return blob_location

def create_scale_space(gray_image,sigma_scale_factor,initial_sigma,level):
    h,w=np.shape(gray_image)
    scale_space = np.zeros((h,w,level),np.float32)
    sigma = [0]*(level+1)
    sigma[0] = initial_sigma
    for i in range(0,level):
        print('Convolving with sigma={}'.format(sigma[i]))
        kernel=laplacian_of_gaussian_filter(sigma[i])
        convolved_image=convolve(gray_image,kernel)
        cv2.imshow("LoG Convolved Image with sigma={}".format(sigma[i]),convolved_image)
        scale_space[:,:,i] = np.square(convolved_image)
        sigma[i+1]=sigma[i]*sigma_scale_factor
    return scale_space,sigma


  
def main():
    #-----------Modify these parameters-----------
    level=14
    threshold_factor=0.02
    initial_sigma=1.35
    sigma_scale_factor=1.24
    #---------------------------------------------
    
    filelocation="TestImages4Project/tiger.jpg"
    image = read_images(filelocation)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.normalize(gray_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    start_time = time.time()
    scale_space,sigma = create_scale_space(gray_image,sigma_scale_factor,
                                           initial_sigma,level) 

    blob_location = max_supression(scale_space,
                                   sigma,
                                   threshold_factor,
                                   level) 
    no_of_blobs=len(blob_location)
    for center in blob_location:
        radius = int(np.ceil(np.sqrt(2)*sigma[center[2]])) 
        cv2.circle(image,(center[1],center[0]),radius, (0,0,255))
    
    print("Number of Blobs=", no_of_blobs)
    print("Total Time Taken (in seconds): {}".format(time.time() - start_time))

    cv2.imshow("Blob Threshold={},Initial Sigma={}, {} circles".format(threshold_factor,initial_sigma, no_of_blobs),image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  


if __name__ == '__main__':
    main()
