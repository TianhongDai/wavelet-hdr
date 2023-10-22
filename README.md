# Wavelet-Based Network For High Dynamic Range Imaging
This is the official code for our paper "Wavelet-Based Network For High Dynamic Range Imaging". The paper is currently under review, please do not distribute the code.

## Requirements
- pytorch==1.4.0
- opencv-python
- scikit-image==0.17.2
- pywavelets==1.1.1

## Datasets
### Kalantari Dataset
Please download the kalantari dataset from this [link](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/).
### RAW Dataset
Please download the kalantari dataset from this [link](https://github.com/TianhongDai/wavelet-hdr) (We are waiting for the approval to upload the dataset). The code for training the RAW dataset is on the [raw branch](https://github.com/TianhongDai/wavelet-hdr/tree/raw).

## Instruction
1. train the network:
```bash
python train.py --cuda
```
2. continue training using the pre-saved checkpoint:
```bash
python train.py --cuda --resume --last-ckpt-path="..."
```
3. test the model and save HDR image:
```bash
python test_psnr.py --cuda --save-image
```
