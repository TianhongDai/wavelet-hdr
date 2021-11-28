# Wavelet-Based Network For High Dynamic Range Imaging
This is the official code for our paper "Wavelet-Based Network For High Dynamic Range Imaging". The paper is currently under review, please do not distribute the code.

## Requirements
- pytorch==1.4.0
- opencv-python
- scikit-image==0.17.2
- pywavelets==1.1.1

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

## BibTex
```
@article{dai2021wavelet,
  title={Wavelet-Based Network For High Dynamic Range Imaging},
  author={Dai, Tianhong and Li, Wei and Cao, Xilei and Liu, Jianzhuang and Jia, Xu and Leonardis, Ales and Yan, Youliang and Yuan, Shanxin},
  journal={arXiv preprint arXiv:2108.01434},
  year={2021}
}
```