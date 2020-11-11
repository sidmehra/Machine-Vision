# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:24:35 2020

@author: sid
"""
# Importing the necessary libraries 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

#----------------------------- Task 1 - FEATURE POINTS------------------------------------- 

#----------------------------- TASK 1 - PART (A)-------------------------------------------

def task1_partA():
    """Return the gray value resized (doubled) image"""
#----------------------------- PART A - Subtask 1------------------------------------------ 
    # Load and read the original input image 
    input_img = cv2.imread('Assignment_MV_01_image_1.jpg')
    # Show the original input image 
    cv2.imshow('Input_image', input_img) 
    # Convert the original image into the grayscale image 
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    #print("Initial data type of gray value image is {}".format(gray_img.dtype))
    print()
    # Show the grayscale image 
    cv2.imshow('GrayValue_image', gray_img)    
    
#----------------------------- PART A - Subtask 2------------------------------------------ 
    # Convert the datatype of image to float 32 to avoid rounding errors 
    gray_img= gray_img.astype(np.float32)
    # Saving the output to local storage after converting to float 32 
    # cv2.imwrite("Gray_Value_Image.png",gray_img)
    
#----------------------------- PART A - Subtask 3------------------------------------------ 
    # Determine the size of gray value image 
    print('Original Shape of the gray value image is {}'.format(gray_img.shape))
    print()
    # Percent by which the input image needs to be resized 
    # Input image needs to be doubled 
    scale_percent= 200 
    # Resize the image to double the size of gray value image 
    width = int(gray_img.shape[1] * scale_percent / 100)
    height = int(gray_img.shape[0] * scale_percent / 100)
    # Define the new dimensions of the image in a tuple 
    dim = (width, height)
    # Call the resize function of the open cv 
    double_img = cv2.resize(gray_img, dim, interpolation = cv2.INTER_AREA)
    # Print the size of the resized image 
    print('Resized shape of the gray value image is {}'.format(double_img.shape))
    # Display the resized image 
    cv2.imshow("Resized image", double_img/np.max(double_img))
    # Saving the output to local storage after resizing the image 
    # cv2.imwrite("Resized_Image.png",double_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    #print("Final data type of the gray value image after conversion is:",gray_img.dtype)
    
    # Return the resized image 
    return double_img 

#------------------------------ Task 1 - PART (B)--------------------------------------------
    
def task1_partB(resized_img):
    """Returns the list of all scale space images at all the scales """
#----------------------------- PART B - Subtask 1------------------------------------------ 
    # List to store all the scale space images at different scales for the futher task 
    scale_space_images=[]
    sigma_list=[]
    for k in range(0,12):
        sigma= 2**(k/2)
        sigma_list.append(sigma)
        # Defining the image grid/ window size to sufficiently capture the characteristic of gaussian 
        x,y = np.meshgrid(np.arange(0,6*sigma),np.arange(0,6*sigma))
        # Defining the gaussian kernel 
        gaussian_kernel= 1/(2*np.pi*sigma**4)*np.exp(-((x-len(x[0])/2)**2+(y-len(x)/2)**2)/(2*sigma**2))
        # Display the 3D plot of the gaussian kernel 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")
        ax.plot_surface(x,y,gaussian_kernel)
        # Display the gaussian kernel as image 
        fig= plt.figure()
        plt.imshow(gaussian_kernel)
        # Saving the plot to the local storage 
        # fig.savefig('smoothing_kernel with k={}.png'.format(k))
        
#----------------------------- PART B - Subtask 2------------------------------------------     
        # Applying the kernel at respective scale to the resized image 
        scale_space_img = cv2.filter2D(resized_img, -1, gaussian_kernel)
        # Store the scale space image at current scale in a list 
        scale_space_images.append(scale_space_img/np.max(scale_space_img))
        # Displaying the scale space image 
        cv2.imshow("Scale Space Image at k={}".format(k),scale_space_img/np.max(scale_space_img))
        # Saving the output to local storage after getting the scale space image at current scale 
        # cv2.imwrite("Scale Space Image at k={}.jpg".format(k),(scale_space_img/np.max(scale_space_img))*255)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    # Return the list of all scale space images 
    return scale_space_images

 #------------------------------ Task 1 - PART (C)-------------------------------------------------
 
def task1_partC(scale_space_images):
    """It returns the list of difference of gaussian images"""
    DoG_images=[]
    for i in range(1,len(scale_space_images)):
        # Calculate different of gaussian at respective scale 
        DoG= scale_space_images[i]-scale_space_images[i-1]
        # Append each DoG image in a list 
        DoG_images.append((DoG/np.max(DoG))*255)
        # Display each DoG Image 
        cv2.imshow('DoG Image {}'.format(i), DoG/np.max(DoG))
        # Saving the output to local storage after computing the difference of gaussians 
        # cv2.imwrite("DoG Image {}.jpg".format(i), (DoG/np.max(DoG))*255)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Return the list of DoG images 
    return DoG_images 


#------------------------------ Task 1 - PART (E)-------------------------------------------------
    
def task1_partE(scale_space_images):
    """Returns the 2 lists of the derivative images at all scales in x and y direction"""
    # Defining the appropriate given kernels in x and y direction 
    dx= np.array([[1, 0, -1]])
    dy= np.array([[1], [0], [-1]])
    # Two list to store the derivative images at all the scales 
    gx_list=[]
    gy_list=[]
    for k in range(0,len(scale_space_images)):
        # Detecting the edges in x and y direction of each scale space image 
        derivative_x= cv2.filter2D(scale_space_images[k], -1, dx)
        gx_list.append(derivative_x)
        derivative_y= cv2.filter2D(scale_space_images[k], -1, dy)
        gy_list.append(derivative_y)
        # Displaying the convoluted image in x and y direction 
        cv2.imshow("Derivative x image at k={}".format(k),derivative_x/np.max(derivative_x))
        cv2.imshow("Derivative y image at k={}".format(k),derivative_y/np.max(derivative_y))
        # Saving the output images 
        # cv2.imwrite("Derivative x image at k={}.jpg".format(k), (derivative_x/np.max(derivative_x))*255)
        # cv2.imwrite("Derivative y image at k={}.jpg".format(k), (derivative_y/np.max(derivative_y))*255)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    
    # Return the 2 lists of Derivative images in x and y direction 
    return gx_list,gy_list
   
#----------------------------- Task 2 - IMAGE MATCHING------------------------------------- 

#----------------------------- TASK 2 - PART (A)-------------------------------------------

def task2_partA():
    """Return the input and target gray value images"""
    # Load, read and display the original input image 
    input_img = cv2.imread('Assignment_MV_01_image_1.jpg') 
    cv2.imshow('Input_image', input_img)
    # Load, read and display the target image 
    target_img = cv2.imread('Assignment_MV_01_image_2.jpg') 
    cv2.imshow('Target_image', target_img)
    # Convert both the images into gray value images 
    input_gray= cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
    #print("The data type of the first image before conversion is {}".format(input_gray.dtype))
    #print("The data type of the second image before conversion is {}".format(target_gray.dtype))
    #print()
    # Converting the data-type of both the images into float32 
    input_gray= input_gray.astype(np.float32)
    target_gray=target_gray.astype(np.float32)
    #print("The data type of the first image after conversion is {}".format(input_gray.dtype))
    #print("The data type of the second image after conversion is {}".format(target_gray.dtype))
    # Displaying the gray value images 
    cv2.imshow("Input Gray Image",input_gray/np.max(input_gray))
    cv2.imshow("Target Gray Image",target_gray/np.max(target_gray))
    # Saving both the input and the target gray value images 
    #cv2.imwrite("Input Gray Image.png",input_gray)
    #cv2.imwrite("Target Gray Image.png",target_gray)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    
    # Return the input and the target original and gray value images 
    return input_img, target_img, input_gray,target_gray

#----------------------------- TASK 2 - PART (B)-------------------------------------------
    
def task2_partB(input_image, coordinates, thickness=5):
    """Return the rectangle image and the window cut out image"""
    # Drawing the rectangle around the window in the input image 
    rectangle_img= cv2.rectangle(input_image, coordinates[0], coordinates[1], (255,0,0), thickness)
    # Display the rectangle image 
    cv2.imshow("Rectangle Image", rectangle_img/np.max(rectangle_img))
    # Saving the rectangle image to the local storage 
    #cv2.imwrite("Rectangle Image.jpg", rectangle_img)
    # Cut out the window image patch 
    x_begin=coordinates[0][0]
    y_begin=coordinates[0][1]
    y_end=coordinates[1][1]
    x_end=coordinates[1][0]
    # Crop the window in the input image through these coordinates 
    image_patch = rectangle_img[y_begin:y_end, x_begin:x_end]
    # Display the cut out window patch 
    cv2.imshow("Cut-out Window Patch Image", image_patch/np.max(image_patch)) 
    # Save the image patch on local storage 
    #cv2.imwrite("Window_Patch.jpg",image_patch)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Return the rectangle image and the window cut out image 
    return rectangle_img, image_patch 

#----------------------------- TASK 2 - PART (C)-------------------------------------------

def task2_partC(cutout_patch):
    # Mean of the cut-out patch 
    cutout_mean= np.mean(cutout_patch)
    # Print the mean of the cut-out patch 
    print("Mean of the cut-out patch is {}".format(cutout_mean))
    print()
    # Standard deviation of the cut-out patch 
    cutout_std= np.std(cutout_patch)
    # Print the standard deviation of the cut-out patch 
    print("Standard deviation of the cut-out patch is {}".format(cutout_std))


def main():
    # Create the task variable to execute the task 1 sub-parts or task 2 sub-parts 
    # To execute task 1 change the variable task = 1
    # To execute the task 2 change the variable task = 2 
    task= 1
    # If task = 1 then execute sub-parts of task 1 else sub-parts of task 2
    if task ==1:
         # Calling the task1_partA function 
         # This function returns the resized input gray value image 
         double_img= task1_partA()
         # Calling the task1_partB function 
         # This function returns the list of all the scale space images 
         scale_space_images= task1_partB(double_img)
         # Calling the task1_partC function 
         # This function returns the list of all DoG images 
         DoG_images= task1_partC(scale_space_images)
         # Calling the task1_partE function 
         gx_list, gy_list= task1_partE(scale_space_images)
         
    elif task==2:
        # Calling the Task 2 part A function 
        input_img, target_img, input_gray,target_gray= task2_partA()
        # Coordinates of the window in the image 
        coordinates= [(360,210),(430,300)]
        # Calling the Task 2 part B function 
        rectangle_img, image_patch= task2_partB(input_gray,coordinates)
        # Calling the Task 2 part C 
        task2_partC(image_patch)
    else:
        print("Incorrect value of the task entered; please enter 1 or 2 into task variable to execute either of tasks")
    
    
# Calling our main function 
main()
    
    
    
    
    
    
    
    
    