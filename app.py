import glob
import os
import subprocess
import sys
import warnings
import zipfile

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import PIL
import streamlit as st

from preprocessing import preprocess

st.header("Photo to Avatar")
st.write("Choose any image and get corresponding avatar:")

uploaded_file = st.file_uploader("Choose an image...")

# newImg.save(out_f)

@st.cache(show_spinner=False)
def download_checkpoint():
    path = './checkpoint/temp.zip'
    if not os.path.exists(path):
        decoder_url = 'wget --no-verbose -O ./checkpoint/temp.zip https://www.dropbox.com/sh/63xqqqef0jtevmg/AADN7izdFHxueUbTSRBZrpffa?dl=0'

        with st.spinner('Downloading pretrained model...'):
            os.system(decoder_url)

        with st.spinner('Unzipping downloaded model...'):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall('./checkpoint/')

        os.rename('./checkpoint/UGATIT_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing',
                    './checkpoint/UGATIT_sample_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing')

# for debugging only:
def get_local_files():
    local_files = []

    for p in glob.glob(pathname="./checkpoint/**", recursive=True):
        if os.path.isfile(p):
            local_files.append(p)

    for p in glob.glob(pathname="./dataset/**", recursive=True):
        if os.path.isfile(p):
            local_files.append(p)

    for p in glob.glob(pathname="./results/**", recursive=True):
        if os.path.isfile(p):
            local_files.append(p)

    return local_files


if uploaded_file is not None:
    download_checkpoint()
    img = PIL.Image.open(uploaded_file).convert("RGB")
    img = np.array(img)
    pre = preprocess.Preprocess()
    # face alignment and segmentation
    face_rgba = pre.process(img)
    face = face_rgba[:,:,:3].copy()
    mask = face_rgba[:,:,3].copy()[:,:,np.newaxis]/255.
    face_white_bg = (face*mask + (1-mask)*255).astype(np.uint8)
    cv2.imwrite('./dataset/sample/testA/0000.png', cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR))

    with st.spinner('Wait for modeling...'):
        # subprocess.run([f"{sys.executable}", "main.py"])
        capture = subprocess.run([f"{sys.executable}", "main.py"], capture_output=True, text=True).stdout

    # st.write("Show output of ML run:")
    # st.text(capture)

    st.markdown("""---""")
    st.write("Show all local ML related files:")
    st.table(get_local_files())
    st.markdown("""---""")

    try:
        img_uploaded = PIL.Image.open(uploaded_file)
    except Exception as e:
        st.error("Loading of uploaded image failed.")
    else:
        st.image(img_uploaded, caption='Input Image', use_column_width=True)

    try:
        img_processed = PIL.Image.open("./dataset/sample/testA/0000.png")
    except Exception as e:
        st.error("Loading of processed image failed. Probably the model did not run.")
    else:
        st.image(img_processed, caption='Processed Image', use_column_width=True)

    try:
        output = PIL.Image.open("./results/UGATIT_sample_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing/0000.png")
    except Exception as e:
        st.error("Loading of result image failed. Probably the model did not run.")
    else:
        st.image(output, caption='Avatar', use_column_width=True)
