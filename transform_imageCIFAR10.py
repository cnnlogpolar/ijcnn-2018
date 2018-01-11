#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:56:27 2017

@author: marta
"""
import glob
import os
import cv2
import numpy as np
import math


def generateimage_Translation():
    directory=  'VOC2006_test/PNGImages/'
    directory_tran= 'VOC2006_trans'
    
    listfile =[]
    
    
    
    
    img_paths = glob.glob(directory + '**/*.png', recursive=True)
    
    for i,filenames in enumerate(img_paths):
        
        
        img = cv2.imread(filenames)
        
        rows,cols,channels = img.shape
        #PASCAL
        M = np.float32([[1,0,20],[0,1,20]])
        
       
        res_trans = cv2.warpAffine(img,M,(cols,rows))

        if os.path.exists(os.path.dirname(os.path.abspath(directory_tran + filenames[os.path.splitext(filenames)[0].find('/'):]))):
            cv2.imwrite(directory_tran + filenames[os.path.splitext(filenames)[0].find('/'):] + ".png",res_trans)
        else:
            print("Crie os diretorios do arquivo: " + directory_tran + filenames[os.path.splitext(filenames)[0].find('/'):] )
            break
            
        #cv2.imshow("Normal", img)
        #cv2.imshow("Rotation 90 neg", res_rot90neg)
        #cv2.imshow("Rotation 90", res_rot90)
        #cv2.imshow("Rotation 180", res_rot180)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

        listfile.append(directory_tran + filenames[os.path.splitext(filenames)[0].find('/'):] + ".png")
        print('read files:' + filenames)

 
        
    return listfile





def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)


def generateimage_Rotation():
    directory=  'TUDarmstadt_test/PNGImages'
    directory_rot= 'TUDarmstadt_test_rot315'
    
    listfile =[]
    
    
    
    
    img_paths = glob.glob(directory + '/**/*.png', recursive=True)
    
    for i,filenames in enumerate(img_paths):
        
        
        img = cv2.imread(filenames)
        
        rows,cols,channels = img.shape

        #res_rot90neg = rotate_about_center(img,-90)
        #cv2.imwrite(os.path.splitext(val)[0] + "_rotneg90.png",res_rot90neg)

        
        
        #res_rot90 = rotate_about_center(img,90)
        #cv2.imwrite(os.path.splitext(val)[0] + "_rot90.png",res_rot90)
        
        
        
        res_rot180 = rotate_about_center(img,315)
        
        if os.path.exists(os.path.dirname(os.path.abspath(directory_rot + filenames[os.path.splitext(filenames)[0].find('/'):]))):
            cv2.imwrite(directory_rot + filenames[os.path.splitext(filenames)[0].find('/'):] ,res_rot180)
        else:
            print("Crie os diretorios do arquivo: " + directory_rot + filenames[os.path.splitext(filenames)[0].find('/'):] )
            break
            
        #cv2.imshow("Normal", img)
        #cv2.imshow("Rotation 90 neg", res_rot90neg)
        #cv2.imshow("Rotation 90", res_rot90)
        #cv2.imshow("Rotation 180", res_rot180)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

        listfile.append(directory_rot + filenames[os.path.splitext(filenames)[0].find('/'):] + ".png")
        print('read files:' + filenames)

 
        
    return listfile





def generateimage_Affine():
    
    directory=  'TUDarmstadt_test/PNGImages'
    directory_aff= 'TUDarmstadt_test_Affine'
    
    listfile =[]
    
    
    img_paths = glob.glob(directory + '/**/*.png', recursive=True)
    
    for i,filenames in enumerate(img_paths):
        
        
        img = cv2.imread(filenames)
        
     
        rows,cols,channels = img.shape
    
        #PASCAL                
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        
    
    
        M = cv2.getAffineTransform(pts1,pts2)
    
        res_affine = cv2.warpAffine(img,M,(cols,rows))
                
        
        if os.path.exists(os.path.dirname(os.path.abspath(directory_aff + filenames[os.path.splitext(filenames)[0].find('/'):]))):
            cv2.imwrite(directory_aff + filenames[os.path.splitext(filenames)[0].find('/'):] + ".png",res_affine)
        else:
            print("Crie os diretorios do arquivo: " + directory_aff + filenames[os.path.splitext(filenames)[0].find('/'):] )
            break
            
        #cv2.imshow("Normal", img)
        #cv2.imshow("Rotation 90 neg", res_rot90neg)
        #cv2.imshow("Rotation 90", res_rot90)
        #cv2.imshow("Rotation 180", res_rot180)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    
        listfile.append(directory_aff + filenames[os.path.splitext(filenames)[0].find('/'):] + ".png")
        print('read files:' + filenames)
    
     
        
    return listfile




def generateimage_Flip():
    directory=  'VOC2006_test/PNGImages/'
    directory_flip= 'VOC2006_Flip'
    #directory='TUDarmstadt/PNGImages/'
    #directory_flip= 'TUDarmstadt_flip'
    
    listfile =[]
    
    
    img_paths = glob.glob(directory + '**/*.png', recursive=True)
    
    for i,filenames in enumerate(img_paths):
        
        
        img = cv2.imread(filenames)

        rimg=cv2.flip(img,1)
                #fimg=cv2.flip(img,0)
                
        
        if os.path.exists(os.path.dirname(os.path.abspath(directory_flip + filenames[os.path.splitext(filenames)[0].find('/'):]))):
            cv2.imwrite(directory_flip + filenames[os.path.splitext(filenames)[0].find('/'):] + ".png",rimg)
        else:
            print("Crie os diretorios do arquivo: " + directory_flip + filenames[os.path.splitext(filenames)[0].find('/'):] )
            break
            
        #cv2.imshow("Normal", img)
        #cv2.imshow("Rotation 90 neg", res_rot90neg)
        #cv2.imshow("Rotation 90", res_rot90)
        #cv2.imshow("Rotation 180", res_rot180)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    
        listfile.append(directory_flip + filenames[os.path.splitext(filenames)[0].find('/'):] + ".png")
        print('read files:' + filenames)
    
     
        
    return listfile


listfile =generateimage_Affine()
