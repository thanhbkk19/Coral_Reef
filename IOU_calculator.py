import cv2
import numpy as np
from transform import linear_transform
import os

def IOU_cal(gt,img):
    total = gt.shape[0]*gt.shape[1]
    correct = np.sum(gt==img)
    return correct/total

if __name__ =="__main__":
    gt= cv2.imread("/home/gumiho/project/coral_reef/jsonfilemask_converted/frame_000000_label_viz.png",0)
    img = linear_transform("/home/gumiho/project/car_racing2/coral_reef/frame_000025.png")
    gt_root = "/home/gumiho/project/coral_reef/jsonfilemask_converted"
    img_root = "/home/gumiho/project/coral_reef/coral_reef"
    
    gt_list = os.listdir(gt_root)
    sample=0
    IOU = 0
    for gt_name in gt_list:
        gt = cv2.imread(os.path.join(gt_root,gt_name),0)
        img_name = gt_name.replace("_label_viz.png",".png")
        img = linear_transform(os.path.join(img_root,img_name))
        IOU += IOU_cal(gt,img)
        sample +=1
    if sample!=0:
        print(IOU/sample)