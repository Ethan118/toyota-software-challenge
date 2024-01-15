import cv2
import sys 
import os 
from PIL import Image 

dir_path = ''


for filename in os.listdir(dir_path ):
    if filename.endswith((".jpg", ".jpeg", ".png", ".gif")):
        image_path = os.path.join(dir_path, filename)

        try:
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #apply parula colormap on depth image
            converted = cv2.applyColorMap(gray_image, cv2.COLORMAP_PARULA)

            #make a new directory for converted images and write conv images into it 
            dir_path_conv = dir_path + '/converted'
            if not os.path.exists(dir_path_conv):
                os.makedirs(dir_path_conv)

            cv2.imwrite(dir_path_conv + '/' + filename, converted)

            width, height = image.size

            print(f"Image: {filename}, Width: {width}, Height: {height}")

    
            image.close()
        except Exception as e:
            print(f"Error processing image: {filename}. Error: {e}")
    else:
        print(f"Ignoring non-image file: {filename}")
