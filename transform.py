import cv2
import numpy as np
def test1():
    img = cv2.imread("/home/gumiho/project/car_racing2/coral_reef/frame_000025.png",1)
    img = cv2.resize(img, (img.shape[1]*2,img.shape[0]*2))
    print(img.shape)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    equ = cv2.equalizeHist(gray)
    fusion = np.hstack((gray,equ))
    cv2.imshow("test",fusion)
    cv2.imshow("real",img)
    cv2.waitKey(0)
def test2():
    img = cv2.imread("/home/gumiho/project/car_racing2/coral_reef/frame_000025.png",1)
    img = cv2.resize(img, (img.shape[1]*2,img.shape[0]*2))
    r = img[:,:,2]
    print(r.shape)
    cv2.imshow("test2",r)
    cv2.waitKey(0)
def test3():
    img = cv2.imread('/home/gumiho/project/car_racing2/coral_reef/frame_000025.png', 1)
    cv2.imshow("img",img) 

    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imshow("lab",lab)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    cv2.imshow('l_channel', l)
    cv2.imshow('a_channel', a)
    cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imshow('final', final)
    cv2.waitKey(0)

def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2


def linear_transform(img_path,binary=True,r1=100,s1=0,r2=210,s2=255):
# Open the image.
    img = cv2.imread(img_path,1)
    #img = cv2.resize(img, (img.shape[1]*2,img.shape[0]*2))
    r = img[:,:,2]  
    equ = cv2.equalizeHist(r)

    # Vectorize the function to apply it to each value in the Numpy array.
    pixelVal_vec = np.vectorize(pixelVal)
    
    # Apply contrast stretching.
    contrast_stretched = pixelVal_vec(equ, r1, s1, r2, s2) 
    # Save edited image.
    contrast_stretched = np.array(contrast_stretched,dtype=np.uint8)
    blank = np.zeros((len(contrast_stretched),len(contrast_stretched[0])))
    for i in range(len(contrast_stretched)):
        for j in range(len(contrast_stretched[0])):

            if contrast_stretched[i][j]==0:
                blank[i][j]=255
                continue
            blank[i][j] = contrast_stretched[i][j]
    blank = np.array(blank,dtype=np.uint8)
    if binary:
        return np.array((blank!=255),dtype=np.uint8)*255.0
    return blank

def binary_img(img):
    img[img==255] = 2
if __name__ =="__main__":
    img = linear_transform("/home/gumiho/project/car_racing2/coral_reef/frame_000025.png")
    cv2.imshow("test",img*255.0)
    cv2.waitKey(0)