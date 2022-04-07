# photo2avatar
Theme: Entertainment and the arts
Objective: Turn photos into a cartoon avatar

## 1. Project Description
1. object detection & segemention (edge detection)
2. discover facial features
3. sketch
4. coloraize


## 2. GitHub Framework

## 3. How to run?

### Pip

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Download
* Put the pre-trained photo2avatar model into checkpoint folder

### Dataset

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

```bash
python main.py --dataset YOUR_DATASET_NAME
```

If the memory of gpu is **not sufficient**, set `--light` to **True**

* But it may **not** perform well

### Test

```bash
python main.py --dataset YOUR_DATASET_NAME --phase test
```

## 4. Code and Paper References
U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation [Paper][Code]

