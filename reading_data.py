import glob
import cv2
import numpy as np
import re

def read_images_in_folder():
    ID=[]
    image_stack = []
    for img in glob.glob('train/*.jpg'): # All jpeg images
        id=re.sub('train/', '', img)
        id=re.sub('.jpg', '', id)
        # print(id)
        image_stack.append(cv2.imread(img))
        ID.append(id)
    return image_stack, ID

def resize_image(image_stack): # Image resizing is performed here
    im_resized_stack = []
    for img in image_stack:
        im_resize = cv2.resize(img, (225, 225), interpolation=cv2.INTER_CUBIC) # Setting image size to 100x100 pixels
        im_resized_stack.append(im_resize)
    return im_resized_stack

if __name__ == '__main__':
    image_stack,ID = read_images_in_folder()
    image_resized_stack = resize_image(image_stack)
    input_train_image=np.array(image_resized_stack)
    ID_list=np.array(ID,dtype=object)
    np.save('train.npy', input_train_image)
    np.save('ID.npy', ID_list)
