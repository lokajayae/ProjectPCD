import tkinter as tk
import numpy as np

from PIL import Image, ImageTk
from tkinter import Menu
from cv2 import cv2
from tkinter import filedialog as fd

kernel = ""
depth = ""
aperture_linear = ""
dialog_box = object()
sigma_color = ""
sigma_space = ""
border_type = ""
matrix_input = ""
matrix_list = []
intensity = ""
the_matrix = [[]]
message = ""

#UTILITY
def rearrange_channel(img_array) :
    b, g, r = cv2.split(img_array)
    return cv2.merge((r, g, b))

def get_binary(message):
    for c in message :
        yield ord(c)

def gcd(x, y) :
    while(y) :
        x, y = y, x % y
    
    return x
    
#CONVOLUTION
def get_convolution_value(windows) :
    prompt_input_convolution(windows)
    windows.wait_window(dialog_box)
    prompt_matrix(windows)
    windows.wait_window(dialog_box)
    return the_matrix, int(depth)

def prompt_input_convolution(windows) :
    global dialog_box
    radioButtonVal = tk.StringVar()
    radioButtonVal.set("3")

    dialog_box = tk.Toplevel(windows)

    def button_clicked() :
        global kernel
        global depth

        kernel = radioButtonVal.get()
        depth = depth_entry.get()
        dialog_box.destroy()
        
    #KERNEL
    kernel_label = tk.Label(dialog_box, text="Kernel : ")
    kernel_label.pack()

    values = {"3x3" : "3",
              "5x5" : "5"}
    for (text, value) in values.items() :
        tk.Radiobutton(dialog_box, text=text, variable=radioButtonVal, value=value).pack()

    #DEPTH
    emptyLabel1= tk.Label(dialog_box, text="")
    emptyLabel1.pack()

    depth_label = tk.Label(dialog_box, text="Depth : ")
    depth_label.pack()
    depth_entry = tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=5)
    depth_entry.pack()

    #BUTTON
    emptyLabel2= tk.Label(dialog_box, text="")
    emptyLabel2.pack()

    button = tk.Button(dialog_box, text="OK", command=button_clicked)
    button.pack()

    dialog_box.gemoetry = ("400x400")

#MEAN
def get_mean_blurring_value(windows) :
    prompt_mean_blurring(windows)
    windows.wait_window(dialog_box)

    return int(kernel)

def prompt_mean_blurring(windows) :
    global dialog_box
    radioButtonVal = tk.StringVar()
    radioButtonVal.set("3")

    dialog_box = tk.Toplevel(windows)
    dialog_box.geometry = ("400x400")

    def button_clicked() :
        global kernel
        global depth

        kernel = radioButtonVal.get()
        dialog_box.destroy()
        
    #KERNEL
    kernel_label = tk.Label(dialog_box, text="Kernel : ")
    kernel_label.pack()

    values = {"3x3" : "3",
              "5x5" : "5"}
    for (text, value) in values.items() :
        tk.Radiobutton(dialog_box, text=text, variable=radioButtonVal, value=value).pack()

    #BUTTON
    emptyLabel2= tk.Label(dialog_box, text="")
    emptyLabel2.pack()

    button = tk.Button(dialog_box, text="OK", command=button_clicked)
    button.pack()

#MEDIAN
def get_median_blurring_value(windows) :
    prompt_median_blurring(windows)
    windows.wait_window(dialog_box)

    return int(aperture_linear)

def prompt_median_blurring(windows) :
    global dialog_box

    dialog_box = tk.Toplevel(windows)
    dialog_box.gemoetry = ("400x400")

    def button_clicked() :
        global aperture_linear

        aperture_linear = aperture_entry.get()
        dialog_box.destroy()

    #DEPTH
    emptyLabel1= tk.Label(dialog_box, text="")
    emptyLabel1.pack()

    aperture_label = tk.Label(dialog_box, text="Aperture : ")
    aperture_label.pack()
    aperture_entry = tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=5)
    aperture_entry.pack()

    #BUTTON
    emptyLabel2= tk.Label(dialog_box, text="")
    emptyLabel2.pack()

    button = tk.Button(dialog_box, text="OK", command=button_clicked)
    button.pack()

#BILATERAL
def get_bilateral_value(windows) :
    prompt_bilateral_filtering(windows)
    windows.wait_window(dialog_box)

    return int(sigma_color), int(sigma_space), int(border_type)

def prompt_bilateral_filtering(windows) :
    global dialog_box

    dialog_box = tk.Toplevel(windows)
    dialog_box.gemoetry = ("400x400")

    def button_clicked() :
        global sigma_color
        global sigma_space
        global border_type

        sigma_color = color_entry.get()
        sigma_space = space_entry.get()
        border_type = border_entry.get()

        dialog_box.destroy()

    #SIGMA COLOR
    emptyLabel1= tk.Label(dialog_box, text="")
    emptyLabel1.pack()

    color_label = tk.Label(dialog_box, text="Sigma Color : ")
    color_label.pack()
    color_entry = tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=5)
    color_entry.pack()

    #SIGMA SPACE
    emptyLabel1= tk.Label(dialog_box, text="")
    emptyLabel1.pack()

    space_label = tk.Label(dialog_box, text="Sigma Space : ")
    space_label.pack()
    space_entry = tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=5)
    space_entry.pack()

    #BORDER TYPE
    emptyLabel1= tk.Label(dialog_box, text="")
    emptyLabel1.pack()

    border_label = tk.Label(dialog_box, text="Aperture : ")
    border_label.pack()
    border_entry = tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=5)
    border_entry.pack()

    #BUTTON
    emptyLabel2= tk.Label(dialog_box, text="")
    emptyLabel2.pack()

    button = tk.Button(dialog_box, text="OK", command=button_clicked)
    button.pack()

#GAUSSIAN
def get_gaussian_value(windows) :
    prompt_gaussian_blurring(windows)
    windows.wait_window(dialog_box)

    return int(kernel), float(standard_deviation)

def prompt_gaussian_blurring(windows) :
    global dialog_box
    radioButtonVal = tk.StringVar()
    radioButtonVal.set("3")

    dialog_box = tk.Toplevel(windows)
    dialog_box.gemoetry = ("400x400")

    def button_clicked() :
        global kernel
        global standard_deviation

        kernel = radioButtonVal.get()
        standard_deviation = sd_entry.get()

        dialog_box.destroy()

    #KERNEL
    kernel_label = tk.Label(dialog_box, text="Kernel : ")
    kernel_label.pack()

    values = {"3x3" : "3",
              "5x5" : "5"}
    for (text, value) in values.items() :
        tk.Radiobutton(dialog_box, text=text, variable=radioButtonVal, value=value).pack()

    #STANDARD DEVIATION
    emptyLabel1= tk.Label(dialog_box, text="")
    emptyLabel1.pack()

    sd_label = tk.Label(dialog_box, text="Standard Deviation : ")
    sd_label.pack()
    sd_entry = tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=5)
    sd_entry.pack()

    #BUTTON
    emptyLabel2= tk.Label(dialog_box, text="")
    emptyLabel2.pack()

    button = tk.Button(dialog_box, text="OK", command=button_clicked)
    button.pack()

#MORPHOLOGY
def get_matrix_kernel(windows) :
    prompt_kernel_size(windows)
    windows.wait_window(dialog_box)
    prompt_matrix(windows)
    windows.wait_window(dialog_box)

    return the_matrix

def prompt_matrix(windows) :
    global dialog_box
    weight_entry = []

    dialog_box = tk.Toplevel(windows)
    dialog_box.gemoetry = ("400x400")

    def button_clicked() :
        global the_matrix
        global matrix_list

        matrix_list.clear()
        for i in range(int(kernel) ** 2) :
            matrix_list.append(float(weight_entry[i].get()))
            
        the_matrix = np.array(matrix_list).reshape(int(kernel), int(kernel))

        dialog_box.destroy()
   
    count = 0
    posx = 20
    posy = 20
    #MATRIX WEIGHT
    for i in range(int(kernel)**2) :
        print(posx, posy)
        weight_entry.append(tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=3))
        weight_entry[i].place(x=posx, y=posy)
        count += 1

        if(count == int(kernel)):
            count = 0
            posx = 20
            posy += 30
        else:
            posx += 30

    posx = (int(kernel) * 30 + 20) / 2
    posy += 20
    #BUTTON-
    button = tk.Button(dialog_box, text="OK", command=button_clicked)
    button.place(x=posx, y=posy)

    dialog_box.geometry(f"{int(kernel) * 30 + 40}x{int(kernel) * 30 + 20 +40}")

#HPF
def get_kernel_size(windows) :
    prompt_kernel_size(windows)
    windows.wait_window(dialog_box)

    return int(kernel)

def prompt_kernel_size(windows) :
    global dialog_box
    radioButtonVal = tk.StringVar()
    radioButtonVal.set("3")

    dialog_box = tk.Toplevel(windows)
    dialog_box.geometry = ("400x400")

    def button_clicked() :
        global kernel

        kernel = radioButtonVal.get()
        dialog_box.destroy()
        
    #KERNEL
    kernel_label = tk.Label(dialog_box, text="Kernel : ")
    kernel_label.pack()

    values = {"3x3" : "3",
              "5x5" : "5"}
    for (text, value) in values.items() :
        tk.Radiobutton(dialog_box, text=text, variable=radioButtonVal, value=value).pack()

    #BUTTON
    emptyLabel2= tk.Label(dialog_box, text="")
    emptyLabel2.pack()

    button = tk.Button(dialog_box, text="OK", command=button_clicked)
    button.pack()
 
#SNP
def get_intensity(windows) :
    prompt_intensity_input(windows)
    windows.wait_window(dialog_box)

    return float(intensity)

def prompt_intensity_input(windows) :
    global dialog_box
    radioButtonVal = tk.StringVar()
    radioButtonVal.set("3")

    dialog_box = tk.Toplevel(windows)
    dialog_box.gemoetry = ("400x400")

    def button_clicked() :
        global intensity

        intensity = sd_entry.get()

        dialog_box.destroy()


    emptyLabel1= tk.Label(dialog_box, text="")
    emptyLabel1.pack()

    sd_label = tk.Label(dialog_box, text="Intensity : ")
    sd_label.pack()
    sd_entry = tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=5)
    sd_entry.pack()

    #BUTTON
    emptyLabel2= tk.Label(dialog_box, text="")
    emptyLabel2.pack()

    button = tk.Button(dialog_box, text="OK", command=button_clicked)
    button.pack()

def get_canny_input(windows) :
    prompt_canny_input(windows)
    windows.wait_window(dialog_box)

    return int(lowt), int(hight)

def prompt_canny_input(windows) :
    global dialog_box

    dialog_box = tk.Toplevel(windows)
    dialog_box.gemoetry = ("400x400")

    def button_clicked() :
        global lowt
        global hight

        lowt = sd_entry.get()
        hight = sd_entry2.get()

        dialog_box.destroy()


    sd_label = tk.Label(dialog_box, text="Low Treshold : ")
    sd_label.pack()
    sd_entry = tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=5)
    sd_entry.pack()
    sd_label2 = tk.Label(dialog_box, text="High Treshold : ")
    sd_label2.pack()
    sd_entry2 = tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=5)
    sd_entry2.pack()

    #BUTTON
    emptyLabel2= tk.Label(dialog_box, text="")
    emptyLabel2.pack()

    button = tk.Button(dialog_box, text="OK", command=button_clicked)
    button.pack()

#OTHER
def get_message(windows) :
    prompt_message_input(windows)
    windows.wait_window(dialog_box)

    return message

def prompt_message_input(windows) :
    global dialog_box

    dialog_box = tk.Toplevel(windows)
    dialog_box.gemoetry = ("400x400")

    def button_clicked() :
        global message

        message = message_entry.get()
        dialog_box.destroy()

    #DEPTH
    emptyLabel1= tk.Label(dialog_box, text="")
    emptyLabel1.pack()

    message_label = tk.Label(dialog_box, text="Message : ")
    message_label.pack()
    message_entry = tk.Entry(dialog_box, font = ('calibre',10,'bold'), width=5)
    message_entry.pack()

    #BUTTON
    emptyLabel2= tk.Label(dialog_box, text="")
    emptyLabel2.pack()

    button = tk.Button(dialog_box, text="OK", command=button_clicked)
    button.pack()