from PIL import Image, ImageEnhance
from pytesseract import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import os
import streamlit as st

st.set_page_config(page_title="INPR", page_icon="🤖")

# page_st = """<style>
#     * {
#       font-family: Comic sans MS;
#     }
# </style>"""

# st.markdown(page_st, unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after{content : "Vinit, Nikunj, Parag, Murtaza";
                         display : block;
                         position: relative;
                         color: #fff4e9;
                         font: san serif;
                         padding: 10px;
                         top:3px;
                         visibility: visible;}
            header {visibility: hidden;}
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Intelligent Number Plate Recognition System")

#Define path to tessaract.exe
# path_to_tesseract = r'C:\Users\Admin\AppData\Local\Tesseract-OCR\tesseract.exe'
path_to_tesseract = r'/usr/bin/tesseract'
#%matplotlib inline
#-----Main------
#Define path to image

# for k in range(2,17):
k = 3
# f_path = os.getcwd()+"/images/" + str(k) + ".jpg"
st.sidebar.title("Select the image")
f_path = st.sidebar.file_uploader("", type=['png', 'jpg'])
#f_path = r"C:\Users\Admin\img\7.jpg"
if f_path is not None:
    # i = cv2.imread(f_path)
    file_bytes = np.asarray(bytearray(f_path.read()), dtype=np.uint8)
    i = cv2.imdecode(file_bytes, 1)
    im = Image.open(f_path)
    enhancer = ImageEnhance.Contrast(im)
    factor = 1.25 #increase contrast
    im = enhancer.enhance(factor)
    im.save('ocr.png',dpi=(300,300))
    image = cv2.imread('ocr.png')

    #---Resizing The Image---
    #image = cv2.resize(image, (500,250))
    #i = cv2.resize(i, (500,250))

    #---Orignal Image---

    # c1, c2, c3 = st.columns(3)
    st.caption("Original Image")
    st.image(i)
    # plt.figure(figsize=(20, 20))
    # plt.subplot(1,4,1)
    # plt.title("Orignal")
    # plt.imshow(i)



    #---Grayscle----
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



    #---Dection---
    lplate_data = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    found = lplate_data.detectMultiScale(image)
    #print(found)
    if len(found)!=0:
        w1 = 0
        h1 = 0
        for (x,y,w,h) in found:
            if w>w1 and h>h1:
                w1 = w 
                x1 = x
                y1 = y
                h1 = h
            else:
                break
        cv2.rectangle(i, (x1,y1), (x1+w1, y1+h1), (0,255,0), 2)
        cv2.rectangle(i, (x1,y1-40), (x1+w1, y1+h1-50), (0,0,0), -1)
        

        #---Cropping---
        image = image[y1:(y1+h1),x1-10:(x1+w1)+10]

    else:
        image = i


    st.caption("Cropped Image")
    st.image(image)
    # plt.subplot(1,4,3)
    # plt.title("Cropped")
    # plt.imshow(image)



    #Extract text from image
    text = pytesseract.image_to_string(image)
    Old = text
    st.sidebar.header("Output:-")
    st.sidebar.write('Old - ',Old,'\n')
    text = ''.join(e for e in text if e.isalnum())
    start = text[0:2]

    if text.isalnum() == True:
        st.sidebar.write('New - ',text)
    else:
        text = pytesseract.image_to_string(i)
        #text = ''.join(e for e in text if e.isalnum())
        st.sidebar.write('New - ',text)
        
    if len(found)!=0:
        w1 = 0
        h1 = 0
        for (x,y,w,h) in found:
            if w>w1 and h>h1:
                w1 = w 
                x1 = x
                y1 = y
                h1 = h
            else:
                break
        cv2.putText(i, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # plt.subplot(1,4, 2)
    # plt.title("Detection")
    # plt.imshow(i)

    st.caption("Detection")
    st.image(i)



    # plt.show()
else:
    st.write("Make sure you image is in JPG/PNG Format.")




