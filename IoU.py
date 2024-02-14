import os
import cv2
import numpy as np
import pandas as pd



def calculate_iou(annotated_imgs_directory, label_imgs_directory):
    IOUs = {}
    annotated_imgs = os.listdir(annotated_imgs_directory)
    label_imgs = os.listdir(label_imgs_directory)

    for label_img_name in label_imgs:
        annotated_img_name = "output_" + label_img_name #change this if annotated image name is different
        annotated_img_path = os.path.join(annotated_imgs_directory, annotated_img_name)
        label_img_path = os.path.join(label_imgs_directory, label_img_name)

        annotated_img = cv2.imread(annotated_img_path)
        label_img = cv2.imread(label_img_path)
        intersection = 0
        union = 0

        if annotated_img is None:
            # print(f"Failed to load images for {annotated_img_name}")
            continue
    
        if label_img is None:
            print(f"Failed to load images for {label_img_name}")
            continue

        height, width, _ = annotated_img.shape  # Assuming both images have the same size
        for y in range(height):
            for x in range(width):
                # Get RGB values for each image
                pixel_annotated_img = annotated_img[y, x]
                pixel_label_img = label_img[y, x]


                # Check if the pixel in the first image is green (0, 255, 0)
                if (pixel_annotated_img == [255, 255, 255]).all() and (pixel_label_img == [255, 255, 255]).all():
                    intersection += 1
                    union += 1
                
                elif (pixel_annotated_img == [255, 255, 255]).all() and (pixel_label_img != [255, 255, 255]).all():
                    union += 1


                elif (pixel_annotated_img != [255, 255, 255]).all() and (pixel_label_img == [255, 255, 255]).all():
                    union += 1

        if intersection == 0 or union == 0:
            continue
        iou = intersection / union

        IOUs[label_img_name[:-4]] = iou
        
    return IOUs       

annotated_imgs_directory = "annotated_images/" #rename to the directory that contains the segmented images with green overlay
label_imgs_directory = "data/MedSAMDemo_2D/labels/" #rename to the directory that contains the labeled data given to us with white overlay"

IOUs = calculate_iou(annotated_imgs_directory, label_imgs_directory)

print(IOUs)

df = pd.DataFrame(list(IOUs.items()), columns=['Label ID', 'IoU'])
excel_file_path = 'IoU.xlsx'
# Check if the file already exists
if os.path.exists(excel_file_path):
    # Write the DataFrame to an Excel file, overwriting it
    df.to_excel(excel_file_path, index=False, mode='w')
    print(f"Data has been overwritten in {excel_file_path}")
else:
    # Write the DataFrame to an Excel file, creating it
    df.to_excel(excel_file_path, index=False)
    print(f"Data has been written to {excel_file_path}")