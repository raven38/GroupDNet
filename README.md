# Semantically Multi-modal Image Synthesis

Unofficical Pytorch implementation of the papaer [Semantically Multi-modal Image Synthesis - Zhen Zhu, Zhiliang Xu, Ansheng You, and Xiang Bai](https://arxiv.org/pdf/2003.12697.pdf) in CVPR 2020. 

[Official impelemntation](https://github.com/Seanseattle/SMIS) will be released soon.


## Instalation

* python 3.7
* pytorch 1.4.0

```
python3 -m pip install -r requirements.txt
```

## Prepare dataest

Download Cityscapes Dataset from (https://www.cityscapes-dataset.com/)

## Train

```
python3 main_train.py --data_root ~/dataset/cityscapes --batch_size 1
```

## Results

Save output images to directory `checkpoint/test/`


