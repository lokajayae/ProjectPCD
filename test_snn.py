import tkinter as tk
import utility as utils
import numpy as np
import random
import math
import copy

from PIL import Image, ImageTk
from tkinter import Menu
from cv2 import cv2
from tkinter import filedialog as fd

kernel = 3

theList = [62, 140, 181, 238, 123, 77, 72, 197, 43, 12, 179, 180, 135, 106, 81, 200, 91, 148, 197, 83, 170, 50, 10, 203, 119]
img = np.array(theList).reshape(5, 5)

print(img)

rows, columns = img.shape
sliding_windows = []
indexer = math.floor(kernel/2)
print(kernel)

output = copy.deepcopy(img)
                
for y in range(rows):        
    for x in range(columns):              
        for win_y in range(y-indexer, y+indexer+1):                   
            for win_x in range(x-indexer, x+indexer+1):      
                if (win_y > -1) and (win_y < rows):
                    if (win_x > -1) and (win_x < columns):
                        sliding_windows.append(img[win_y, win_x])  
                

        rate = 0
        if(len(sliding_windows) == (kernel*kernel)):
            #N_S
            if((img[y, x] in range (sliding_windows[1], sliding_windows[7])) or (img[y, x] in range (sliding_windows[7], sliding_windows[1]))) :
                n_s = min([sliding_windows[1], sliding_windows[7]], key=lambda nums:abs(nums-img[y, x]))
                rate += n_s
            else :
                rate += img[y,x]

            #W_E
            if((img[y, x] in range (sliding_windows[3], sliding_windows[5])) or (img[y, x] in range (sliding_windows[5], sliding_windows[3]))) :
                w_e = min([sliding_windows[5], sliding_windows[3]], key=lambda nums:abs(nums-img[y, x]))
                rate += w_e
            else :
                rate += img[y,x]

            #SE_NW
            if((img[y, x] in range (sliding_windows[0], sliding_windows[8])) or (img[y, x] in range (sliding_windows[8], sliding_windows[0]))) :
                nw_se = min([sliding_windows[1], sliding_windows[7]], key=lambda nums:abs(nums-img[y, x]))
                rate += nw_se
            else :
                rate += img[y,x]

            #SW_NE
            if((img[y, x] in range (sliding_windows[2], sliding_windows[6])) or (img[y, x] in range (sliding_windows[6], sliding_windows[2]))) :
                sw_ne = min([sliding_windows[2], sliding_windows[6]], key=lambda nums:abs(nums-img[y, x]))
                rate += sw_ne
            else :
                rate += img[y,x]

            output[y, x] = int(rate/4)
                        
            sliding_windows.clear()
        else:
            sliding_windows.clear()

print(output)