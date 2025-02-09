import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
from tqdm import tqdm

class_names = ['lymphocyte','monocyte']

def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border
	
""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes
	
def create_sub_masks(mask_image, classes):
    height, width = mask_image.shape
    sub_masks = {}
    for i in range(1, classes+1):
      img = np.zeros([width, height],dtype=np.uint8)
      for x in range(width):
        for y in range(height):
          if mask_image[x][y] == i:
            img[x][y] = 255
      sub_masks[i] = img
    return sub_masks
	
def getjson(path, savepath, str1, classes):
   #tile_width = 512
   coco = Coco()
   file_count = sum(len(files) for _, _, files in os.walk(path))
   print(file_count)
   with tqdm(total=file_count) as pbar:
    for p,d,f in os.walk(path):
     for f1 in f: 
        impath = p+f1
        labelspath = '/'.join(path.split('/')[:-1])+'_labels'+'/'+f1.split('.')[0]+'.txt'
        
        width, height = Image.open(impath).size
        tile_width = width
        coco_image = CocoImage(file_name=impath, height=height, width=width)
        fl = open(labelspath, 'r')
        for fll in fl:
          #print(fll)
          class_ = int(fll.split(' ')[0])
          if class_ < 2:
           bbox = fll.split('\n')[0].split(' ')[1:]
           #print(bbox, class_)
           '''coco_image.add_annotation(
                CocoAnnotation(
                bbox=[max(0,(tile_width*float(bbox[0])-18)),max(0,(tile_width*float(bbox[1])-18)),400*float(bbox[2]),400*float(bbox[3])],
                category_id=class_,
                category_name=class_names[class_]
                )
             )'''
           # Calculate initial x, y, width, and height
           #x = tile_width * float(bbox[0]) - 25
           #y = tile_width * float(bbox[1]) - 25
       
           #if bbox[2]==0.1:
           x = tile_width * float(bbox[0]) - 25
           y = tile_width * float(bbox[1]) - 25
           w = 50 #0  * float(bbox[2])
           h = 50 #0  * float(bbox[3])
           #else:
           #    x = tile_width * float(bbox[0]) - 20
           #    y = tile_width * float(bbox[1]) - 20
           #    w = 40 #0  * float(bbox[2])
           #    h = 40 #0  * float(bbox[3])


           # Adjust x and width if x is negative
           if x < 0:
                w += x  # Reduce width by the overshoot
                x = 0   # Clamp x to 0

           # Adjust y and height if y is negative
           if y < 0:
            h += y  # Reduce height by the overshoot
            y = 0   # Clamp y to 0

           # Ensure the bounding box does not exceed image dimensions
           if x + w > tile_width:
                w = tile_width - x
           if y + h > tile_width:
                h = tile_width - y

           # Add annotation
           coco_image.add_annotation(
                CocoAnnotation(
                bbox=[x, y, max(0, w), max(0, h)],  # Ensure width and height are non-negative
                category_id=class_,
                category_name=class_names[class_]
                )
            )
        coco.add_image(coco_image)
        fl.close()
        pbar.update(1)
    for i in range(0,classes-1):
      coco.add_category(CocoCategory(id=i, name=class_names[i]))
    save_json(data=coco.json, save_path=savepath)
    
getjson('/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/g_data/40x/Iter1/train/', '/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/g_data/40x/Iter1/json/train.json','train', 3)
getjson('/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/g_data/40x/Iter1/val/', '/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/g_data/40x/Iter1/json/val.json','val', 3)
getjson('/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/g_data/40x/Iter1/val-cpg/', '/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/g_data/40x/Iter1/json/val_cfg.json','val', 3)
#getjson('/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/g_data/40x/Iter2/val-cpg/', '/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/g_data/20x/Iter2/val-cpg.json','val', 3)
#getjson('/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/g_data/20x/Iter/temp/', '/mnt/prj002/Monkey_Challenge/Monkey_Challenge_Dataset/g_data/20x/Iter/temp.json','temp', 3)
