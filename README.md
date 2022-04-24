# photo2avatar
Theme: Entertainment and the arts
Objective: Turn photos into a cartoon avatar

## 1. Project Description
Novelty:  
Limit & Futher Work: 


## 2. Model Structure


## 3. How to run?

### Setup

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Dataset

* Selfie Dataset - https://www.crcv.ucf.edu/data/Selfie
* Avatar Dataset - https://www.gwern.net/Danbooru2021

* You can also download a dataset [sample](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view) to train and test the model.


### Dataset Structure
The datasets you decide to use should be put into the dataset folder of the project and follow the below structure: 

```bash
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format does not matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg
           ├── ddd.png
           └── ...
```

### Train

```python
python main.py --dataset YOUR_DATASET_NAME
```

If the memory of gpu is **not sufficient**, set `--light` to **True**. But it may **not** perform well

There are a variety of different parameters that can also be passed in. They can be found at the top of `main.py`; however I've pasted a couple important ones below

```python
parser.add_argument('--phase',      default='test',        help='[train / test]')
parser.add_argument('--light',      default=False,          help='[full version / light version]')
parser.add_argument('--dataset',    default='sample',       help='dataset_name')
parser.add_argument('--epoch',      default=100,            help='The number of epochs to run')
parser.add_argument('--iteration',  default=10000,          help='The number of training iterations')
parser.add_argument('--batch_size', default=1,              help='The size of batch size')
parser.add_argument('--print_freq', default=1000,           help='The number of image_print_freq')
parser.add_argument('--save_freq',  default=1000,           help='The number of ckpt_save_freq')
parser.add_argument('--decay_flag', default=True,           help='The decay_flag')
parser.add_argument('--decay_epoch',default=50,             help='decay epoch')
```

### Resume train 

Download the pretrined model [checkpoint](https://www.dropbox.com/sh/63xqqqef0jtevmg/AADN7izdFHxueUbTSRBZrpffa?dl=0) and place it into the checkpoint folder.

The information about the model are as following:

```
##### Information #####
# light :  False
# gan type :  lsgan
# dataset :  sample
# max dataset number :  0
# batch_size :  1
# epoch :  100
# iteration per epoch :  10000
# smoothing :  True

##### Generator #####
# residual blocks :  4

##### Discriminator #####
# discriminator layer :  6
# the number of critic :  1
# spectral normalization :  True

##### Weight #####
# adv_weight :  1
# cycle_weight :  10
# identity_weight :  10
# cam_weight :  1000
```

### Test

```python
python main.py --dataset YOUR_DATASET_NAME --phase test
```

### Colab Example
Open and run the Jupyter Notebook: ```Test.ipynb```

### Deployment application
Run and execute the streamlit app for applications demo: ```streamlit run app.py```

## 4. Tips
* For the errors ```No module named 'tensorflow.contrib'```, it is because version 2.0 of Tensorflow isn't supported, you might need to ```pip uninstall tensorflow==2.8.0``` and then ```pip install tensorflow-gpu==1.14```
    
## 5. Code and Paper References
* U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation [Paper](https://arxiv.org/abs/1907.10830) | [Code](https://github.com/taki0112/UGATIT)
* Minivision's photo-to-cartoon translation project [Code](https://github.com/minivision-ai/photo2cartoon/blob/master/README_EN.md)

