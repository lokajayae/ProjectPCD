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

#variabel global
filename = ""
img_array = [[]]
label = object()
img_tk = object()

original_label = object()
original_pic = object()
output_label = object()
output_pic = object()

def display() :
    windows = tk.Tk()
    windows.geometry("300x300")
    windows.title("IP 181524007")

    canvas = tk.Canvas(windows, width=300, height=300)
    canvas.pack(expand=True)

    deploy_components(windows, canvas)

    windows.mainloop()

def deploy_components(windows, canvas) :
    #define callback
    def plot_original(i_img_array, desc):
        global original_label
        global original_pic
        global output_label
        global output_pic

        canvas.delete(original_label)
        canvas.delete(original_pic)
        canvas.delete(output_label)
        canvas.delete(output_pic)

        if(len(i_img_array.shape) == 3) :
            im = Image.fromarray(utils.rearrange_channel(i_img_array))
        else :
            im = Image.fromarray(i_img_array)

        width, height = im.size
        
        #ImageTk from Image
        global img_tk
        img_tk = ImageTk.PhotoImage(master=windows, image=im)

        original_label = canvas.create_text(30+width/2, 40, text=desc)
        original_pic = canvas.create_image(30+width/2, 50+height/2, image=img_tk)

        canvas.update()

        w = width*2 + 125
        h = height +100

        windows.geometry(f"{width*2 + 125}x{height +100}")
        canvas.config(width=w, height=h)

    def plot_output(i_img_array, desc):
        #Image from Array
        if(len(i_img_array.shape) == 3) :
            im = Image.fromarray(utils.rearrange_channel(i_img_array))
        else :
            im = Image.fromarray(i_img_array)

        width, height = im.size
        
        #ImageTk from Image
        global img_tk_out
        img_tk_out = ImageTk.PhotoImage(master=windows, image=im)
        
        #Label from ImageTk
        global output_label
        global output_pic
        
        output_label = canvas.create_text(50 + 25 + width +width/2, 40, text=desc)
        output_pic = canvas.create_image(50 + 25 + width + width/2, 50+height/2, image=img_tk_out)

        canvas.update()

    def plot_text(text, desc, width, height) :
        global output_label
        global output_pic

        output_label = canvas.create_text(50 + 25 + width +width/2, 40, text=desc)
        output_pic = canvas.create_text(50 + 25 + width + width/2, 50+height/2, text=text)

        canvas.update()

    def open_file() :
        global img_array
        global label 
        global filename

        #if filename :
            #label.pack_forget()

        filename = fd.askopenfilename(initialdir="/asset/", 
                                      title="Open an Image", 
                                      filetypes=(("PNG Files", "*.png"), ("JPG Files", "*.jpg"), ("All Files", "*.*")))

        img_array = cv2.imread(filename)
        plot_original(img_array, "Original")

    def save_file() :
        cv2.imwrite(filename, img_array)
        return True

    def save_as() :
        files = [('PNG Files', '*.png'),
                 ('JPG Files' , '*.jpg'),
                 ('All Files', '*.*')]
        new_filename = fd.asksaveasfilename(filetypes=files, defaultextension=files)
        cv2.imwrite(new_filename, img_array)
        return True

    def convolution() :
        global img_array

        plot_original(img_array, "Original")

        theKernel, depth = utils.get_convolution_value(windows)

        img_array = cv2.filter2D(img_array, depth, theKernel)

        plot_output(img_array, "Convolution")
        return True

    def bilateral() :
        global img_array
        
        plot_original(img_array, "Original")

        color, space, border = utils.get_bilateral_value(windows)

        img_array = cv2.bilateralFilter(img_array, color, space, border)

        plot_output(img_array, "Bilateral Filtering")
        return True

    def mean_blurring() :
        global img_array

        plot_original(img_array, "Original")

        kernel = utils.get_mean_blurring_value(windows)

        img_array = cv2.blur(img_array, (kernel, kernel))

        plot_output(img_array, "Mean Filtering")

    def gaussian_blurring() :
        global img_array

        plot_original(img_array, "Original")

        kernel, standard_deviation = utils.get_gaussian_value(windows)
        img_array = cv2.GaussianBlur(img_array, (kernel, kernel), standard_deviation)

        plot_output(img_array, "Gaussian Filtering")
        return True

    def median_blurring() :
        global img_array

        plot_original(img_array, "Original")

        aperture = utils.get_median_blurring_value(windows)
        img_array = cv2.medianBlur(img_array, aperture)

        plot_output(img_array, "Median Filtering")
        return True

    def laplacian() :
        global img_array

        plot_original(img_array, "Original")

        the_kernel = utils.get_matrix_kernel(windows)

        img_array = cv2.filter2D(img_array, -1, the_kernel)

        plot_output(img_array, "Laplacian Filtering")

    def minmax() :
        global img_array

        kernel = utils.get_kernel_size(windows)

        if(len(img_array.shape) == 3):
            img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else :
            img = copy.deepcopy(img_array)

        plot_original(img, "Original")

        output = img
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
                

                if(len(sliding_windows) == (kernel*kernel)):
                    sliding_windows.pop(math.ceil((kernel*kernel) / 2) - 1)

                    max_value = max(sliding_windows)
                    min_value = min(sliding_windows)
                    rate = (max_value - min_value)/2

                    if img[y, x] > rate:      
                        output[y, x] = max_value
                    elif img[y, x] < rate:
                        output[y, x] = min_value
                        
                    sliding_windows.clear()
                else:
                    sliding_windows.clear()

        img_array = copy.deepcopy(output)
        plot_output(img_array, "MinMax")

    def snn() :
        global img_array

        kernel = 3
        
        if(len(img_array.shape) == 3):
            img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else :
            img = copy.deepcopy(img_array)

        plot_original(img, "Original")

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

        img_array = copy.deepcopy(output)
        plot_output(img_array, "SNN")

    def conservative() :
        global img_array

        kernel = utils.get_kernel_size(windows)

        if(len(img_array.shape) == 3):
            img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else :
            img = copy.deepcopy(img_array)

        plot_original(img, "Original")

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

                if(len(sliding_windows) == (kernel*kernel)):
                    sliding_windows.pop(math.ceil((kernel*kernel) / 2) - 1)

                    max_value = max(sliding_windows)
                    min_value = min(sliding_windows)
                    rate = (max_value - min_value)/2

                    if img[y, x] > rate:      
                        output[y, x] = max_value
                    elif img[y, x] < rate:
                        output[y, x] = min_value
                        
                    sliding_windows.clear()
                else:
                    sliding_windows.clear()

        img_array = copy.deepcopy(output)
        plot_output(img_array, "Conservative")

    def salt_and_pepper() :
        global img_array

        img = copy.deepcopy(img_array)

        plot_original(img, "Original")

        rows, columns, channel = img.shape
        intensity = utils.get_intensity(windows)

        output = np.zeros(img.shape, np.uint8)

        for i in range(rows):
            for j in range(columns):
                r = random.random()

                if r < intensity/2 :
                    #pepper
                    output[i][j] = [0, 0, 0]
                elif r < intensity :
                    #salt
                    output[i][j] = [255, 255, 255]
                else : 
                    output[i][j] = img[i][j]

        img_array = copy.deepcopy(output)
        plot_output(img_array, "Salt n Pepper")

    def gauss_noise() :
        global img_array

        img = img_array

        if(len(img.shape) == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else :
            img = copy.deepcopy(img_array)

        plot_original(img, "Original")

        rows, columns = img.shape

        gauss = np.random.normal(0, 0.5, (rows, columns))
        gauss = gauss.reshape(rows, columns)

        output = cv2.add(img, gauss.astype('uint8'))
        img_output = Image.fromarray(output)

        img_array = np.asarray(img_output)
        plot_output(img_array, "Gaussian Noise")

    def exponential_noise() :
        global img_array

        plot_original(img_array, "Original")

        img = copy.deepcopy(img_array)

        if(len(img.shape) == 3):
            img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gs = copy.deepcopy(img)

        rows, columns = img_gs.shape

        exponential = np.random.exponential(3.45, (rows, columns))
        exponential = exponential.reshape(rows, columns)

        output = cv2.add(img_gs, exponential.astype('uint8'))
        img_output = Image.fromarray(output)

        img_array = np.asarray(img_output)
        plot_output(img_array, "Exponential Noise")

    def speckle_noise() :
        global img_array

        img = copy.deepcopy(img_array)

        plot_original(img, "Original")

        rows, columns, channel = img.shape
        gauss = np.random.randn(rows, columns, channel)
        gauss = gauss.reshape(rows, columns, channel)

        output = img + (img*gauss)
        img_output = Image.fromarray(output.astype('uint8'))

        img_array = np.asarray(img_output)
        plot_output(img_array, "Speckle")

    def poisson() :
        global img_array

        plot_original(img_array, "Original")

        vals = len(np.unique(img_array))
        vals = 2 ** np.ceil(np.log2(vals))
        output = np.random.poisson(img_array * vals) / float(vals)

        img_output = Image.fromarray(output.astype('uint8'))

        img_array = np.asarray(img_output)
        
        plot_output(img_array, "Poisson Noise")

    def erosion() :
        global img_array
        plot_original(img_array, "Original")

        kernel = utils.get_matrix_kernel(windows)

        img_array = cv2.erode(img_array, kernel, iterations=1)
        
        plot_output(img_array, "Erosion")
    
    def dilation() :
        global img_array

        plot_original(img_array, "Original")

        kernel = utils.get_matrix_kernel(windows)

        img_array = cv2.dilate(img_array, kernel, iterations=1)
        
        plot_output(img_array, "Dilation")

    def opening() :
        global img_array

        plot_original(img_array, "Original")

        kernel = utils.get_matrix_kernel(windows)

        img_array = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)

        plot_output(img_array, "Opening")

    def closing() :
        global img_array
        plot_original(img_array, "Original")
        kernel = utils.get_matrix_kernel(windows)
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        plot_output(img_array, "Closing")

    def canny():
        global img_array
        if(len(img_array) == 3):
            img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else :
            img = copy.deepcopy(img_array)
        
        lowt, hight = utils.get_canny_input(windows)
        plot_original(img, "Original")
        img_array = cv2.Canny(img, lowt, hight)
        plot_output(img_array, "Canny")

    def sobel() :
        global img_array
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S

        img = cv2.GaussianBlur(img_array, (3, 3), 0)

        if(len(img.shape) == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        img_array = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        plot_output(img_array, "Sobel Edge Detection")

    def prewitt():
        global img_array

        if(len(img_array.shape) == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(img, -1, kernelx)
        img_prewitty = cv2.filter2D(img, -1, kernely)

        img_array = cv2.add(img_prewittx, img_prewitty)
        plot_output(img_array, "Prewitt Edge Detection")
    
    def robert():
        global img_array

        img = cv2.GaussianBlur(img_array, (3, 3), 0)
        if(len(img.shape) == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernelx = np.array([[1,0],[0,-1]])
        kernely = np.array([[0,-1],[1,0]])
        img_robertx = cv2.filter2D(img, -1, kernelx)
        img_roberty = cv2.filter2D(img, -1, kernely)

        img_array = cv2.add(img_robertx, img_roberty)
        plot_output(img_array, "Robert Edge Detection")

    def encode_stego_text() :
        global img_array

        img = copy.deepcopy(img_array)
        output = copy.deepcopy(img_array)

        plot_original(img, "Original")

        message = utils.get_message(windows)
        iter_msg = utils.get_binary(message)
        msg_unicode = list(iter_msg)
        msg_len = len(msg_unicode)
        idx = 0
        rows, columns, channel = img.shape

        pattern = utils.gcd(rows, columns)

        for i in range(rows) :
            for j in range(columns) :
                if (i+1 * j+1) % pattern == 0:
                    if idx != msg_len :
                        output[i-1][j-1][0] = msg_unicode[idx]
                        idx += 1
                    else :
                        output[i-1][j-1][0] = 0

        img_array = copy.deepcopy(output)
        plot_output(img_array, "Message Embedded")

    def decode_stego_text() :
        global img_array

        img = copy.deepcopy(img_array)
        plot_original(img, "Original")
        rows, columns, channel = img.shape      
        width, height = Image.fromarray(img).size

        pattern = utils.gcd(rows, columns)
        message = ''

        for i in range(rows) :
            for j in range(columns) :
                if(i+1 * j+1) % pattern == 0 :
                    if img[i-1][j-1][0] != 0 :
                        message = message + chr(img[i-1][j-1][0])
                    else :
                        break

        plot_text(message, "Secret Message", width, height)

    def encode_stego_image() :
        global img_array

        img = copy.deepcopy(img_array)
        
        filename = fd.askopenfilename(initialdir="/asset/", 
                                      title="Choose Image to be Hidden", 
                                      filetypes=(("PNG Files", "*.png"), ("JPG Files", "*.jpg"), ("All Files", "*.*")))
        hidden_img = cv2.imread(filename)

        rows, columns, channel = hidden_img.shape

        for i in range(rows) :
            for j in range(columns) :
                for ch in range(channel) :
                    #v1 and v2 are 8-bit pixel values
                    #of img and hidden_img
                    v1 = format(img[i][j][ch], '08b')
                    v2 = format(hidden_img[i][j][ch], '08b')

                    #4 MSBs from each image
                    v3 = v1[:4] + v2[:4]

                    img[i][j][ch] = int(v3, 2)

        img_array = copy.deepcopy(img)
        plot_output(img_array, "Image Hidden")
        return True

    def decode_stego_image() :
        global img_array

        rows, columns, channel = img_array.shape

        original_image = np.zeros((rows, columns, channel), np.uint8)
        hidden_image = np.zeros((rows, columns, channel), np.uint8)

        for i in range(rows) :
            for j in range(columns) :
                for ch in range(channel) :
                    v1 = format(img_array[i][j][ch], '08b')
                    v2 = v1[:4] + chr(random.randint(0, 1)+48) * 4
                    v3 = v1[4:] + chr(random.randint(0, 1)+48) * 4

                    original_image[i][j][ch] = int(v2, 2)
                    hidden_image[i][j][ch] = int(v3, 2)
        
        img_array = copy.deepcopy(original_image)
        plot_original(original_image, "Original Image")
        plot_output(hidden_image, "Hidden Image")

    def  segmentation_kmeans():
        global img_array

        img = img_array.reshape(-1, 3)
        img = np.float32(img)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # atetempts, k = utils.get_cluster()
        k = 3
        attempts = 10

        ret, label, center = cv2.kmeans(img, k, None, criteria, attempts,  cv2.KMEANS_PP_CENTERS )
        
        #convert to uint8
        center = np.uint8(center)
        res = center[label.flatten()]
        output = res.reshape((img_array.shape))

        img_array = copy.deepcopy(output)
        plot_output(output, "K-Means Segmentation")
        return True

    # Creating Menubar
    menubar = Menu(windows) 
    
    # Adding File Menu and commands 
    file = Menu(menubar, tearoff = 0) 
    menubar.add_cascade(label ='File', menu = file) 
    file.add_command(label ='Open File', command = open_file) 
    file.add_command(label ='Save As', command = save_as) 
    file.add_command(label ='Save', command = save_file) 
    file.add_separator() 
    file.add_command(label ='Exit', command = windows.destroy)

    # Adding Edit menu and commands
    edit = Menu(menubar, tearoff=0)
    menubar.add_cascade(label='Edit', menu=edit)

    filtering = Menu(edit, tearoff=0)
    edit.add_cascade(label='Filtering', menu = filtering)
    filtering.add_command(label='Convolution', command=convolution) #done
    filtering.add_command(label='Bilateral', command=bilateral) #done
    filtering.add_command(label='Laplacian Sharpening', command=laplacian) #done
    filtering.add_command(label='MinMax', command=minmax) #need test
    filtering.add_command(label='SNN', command=snn)
    filtering.add_command(label='Conservative', command=conservative) #need test
    filtering.add_command(label='Mean Blurring', command=mean_blurring) #done
    filtering.add_command(label='Gaussian Blurring', command=gaussian_blurring) #done
    filtering.add_command(label='Median Blurring', command=median_blurring) #done

    noise = Menu(edit, tearoff=0)
    edit.add_cascade(label='Noise', menu=noise)
    noise.add_command(label='Salt and Pepper', command=salt_and_pepper) #done
    noise.add_command(label='Impulse Exponential', command=exponential_noise)
    noise.add_command(label='Impulse Gaussian', command=gauss_noise)
    noise.add_command(label='Speckle', command=speckle_noise) #done
    #noise.add_command(label='Poisson', command=poisson) 

    morphological = Menu(edit, tearoff=0)
    edit.add_cascade(label='Morphological', menu=morphological)
    morphological.add_command(label='Dilation', command=dilation) #done
    morphological.add_command(label='Erosion', command=erosion) #done
    morphological.add_command(label='Opening', command=opening) #done
    morphological.add_command(label='Closing', command=closing) #done

    edgy = Menu(edit, tearoff=0)
    edit.add_cascade(label="Edge Detection", menu=edgy)
    edgy.add_command(label="Canny", command=canny)
    edgy.add_command(label="Sobel", command=sobel)
    edgy.add_command(label="Prewitt", command=prewitt)
    edgy.add_command(label="Robert", command=robert)

    stego = Menu(edit, tearoff=0)
    text_stego = Menu(stego, tearoff=0)
    img_stego = Menu(stego, tearoff=0)
    
    edit.add_cascade(label="Steganography", menu=stego)
    stego.add_cascade(label="Text", menu=text_stego)
    stego.add_cascade(label="Image", menu=img_stego)

    text_stego.add_command(label="Encode", command = encode_stego_text)
    text_stego.add_command(label="Decode", command = decode_stego_text)
    img_stego.add_command(label="Encode", command = encode_stego_image)
    img_stego.add_command(label="Decode", command = decode_stego_image)

    segmentation = Menu(edit, tearoff=0)
    edit.add_cascade(label="Segmentation", menu=segmentation)
    segmentation.add_command(label="K-Means", command=segmentation_kmeans)
    segmentation.add_command(label="Split", command=None)
    segmentation.add_command(label="Merge", command=None)
    segmentation.add_command(label="Active Contour", command=None)
    
    windows.config(menu = menubar)