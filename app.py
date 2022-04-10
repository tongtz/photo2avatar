import streamlit as st
from numpy import load
from numpy import expand_dims
from PIL import Image
import os
import zipfile
import cv2
import numpy as np
import preprocessing
from preprocessing import preprocess
import UGATIT
import urllib.request
import dlib
import tensorflow as tf
import argparse
import subprocess
import sys

st.header("Photo to Avatar")
st.write("Choose any image and get corresponding avatar:")

uploaded_file = st.file_uploader("Choose an image...")

# newImg.save(out_f)

# @st.cache
def download_checkpoint():
    
    path = './checkpoint/temp'
	
    if not os.path.exists(path):

        url = 'https://www.dropbox.com/sh/63xqqqef0jtevmg/AADN7izdFHxueUbTSRBZrpffa?dl=0'

        urllib.request.urlretrieve(url, './checkpoint/temp')
  
        with zipfile.ZipFile(path, 'r') as zip_ref:
          zip_ref.extractall('./checkpoint/UGATIT_sample_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing')
	
	
if uploaded_file is not None:
    download_checkpoint()
	
    # src_image = load_image(uploaded_file)
    img = Image.open(uploaded_file)	
	
    img = np.array(img)
    # img = torch.from_numpy(img).type(torch.FloatTensor) 
    pre = preprocess.Preprocess()

    # face alignment and segmentation
    face_rgba = pre.process(img)
    if face_rgba is not None:
    # change background to white
      face = face_rgba[:,:,:3].copy()
      mask = face_rgba[:,:,3].copy()[:,:,np.newaxis]/255.
      face_white_bg = (face*mask + (1-mask)*255).astype(np.uint8)
      face_white_bg = cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR)
      cv2.imwrite(os.path.join('./dataset/sample/testA','.png'), cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR))


    subprocess.run([f"{sys.executable}", "main.py --dataset sample --phase test"])
	
	
    img_processed = Image(filename="./dataset/sample/testA/0000.png")
    output = Image(filename="./results/UGATIT_sample_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/0000.png")
	
    st.image(uploaded_file, caption='Input Image', use_column_width=True)
    #st.write(os.listdir())
    st.image(img_processed, caption='Processed Image', use_column_width=True) 
    st.image(output, caption='Avatar', use_column_width=True) 
	
	
