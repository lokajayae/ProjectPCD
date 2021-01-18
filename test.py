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

img = np.array([132, 232, 72, 66, 173, 102, 206, 102, 226, 134, 97, 187, 67, 160, 190, 130, 220, 64, 100, 65, 198, 125, 78, 132, 187]).reshape(5, 5)
kernel = 3

print(img)

rows, columns = img.shape
sliding_windows = []
indexer = math.floor(kernel/2)

output = copy.deepcopy(img)
        
for y in range(rows):        
    for x in range(columns):              
        for win_y in range(y-indexer, y+indexer+1):                   
            for win_x in range(x-indexer, x+indexer+1):      
                if (win_y > -1) and (win_y < rows):
                    if (win_x > -1) and (win_x < columns):
                        sliding_windows.append(img[win_y, win_x])  
        
            if(sliding_windows):
                print(sliding_windows)

                max_value = max(sliding_windows)
                min_value = min(sliding_windows)
                rate = (max_value - min_value)/2

                if img[y, x] > rate:      
                    output[y, x] = max_value
                elif img[y, x] < rate:
                    output[y, x] = min_value
                    
                sliding_windows.clear()

print(output)