# Wavelet-Based Network For High Dynamic Range Imaging
This is the official code for our paper "[Wavelet-Based Network For High Dynamic Range Imaging](https://www.sciencedirect.com/science/article/pii/S1077314223002618)" [CVIU 2023].
![netowrk_structure](assets/FHDRNet.pdf)
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

## Pretrained Model
Please download the pretained model from this [link](https://github.com/TianhongDai/wavelet-hdr/releases/tag/v1.0.0) or you can use the following command:
```bash
wget https://github.com/TianhongDai/wavelet-hdr/releases/download/v1.0.0/model.pt.zip
unzip model.pt.zip
```

## Instruction
1. train the network:
```bash
python train.py --cuda --use-bn --dataset-path <path-to-training-set> --testset-path <path-to-test-set>
```
2. continue training using the pre-saved checkpoint:
```bash
python train.py --cuda --use-bn --resume --last-ckpt-path <ckpt-path>
```
3. test the model and save HDR image:
```bash
python test.py --cuda --use-bn --save-path <model-path> --save-image
```
## BibTex
To cite this code for publications - please use:
```
@article{dai2024wavelet,
  title={Wavelet-based Network for High Dynamic Range Imaging},
  author={Dai, Tianhong and Li, Wei and Cao, Xilei and Liu, Jianzhuang and Jia, Xu and Leonardis, Ales and Yan, Youliang and Yuan, Shanxin},
  journal={Computer Vision and Image Understanding},
  volume={238},
  pages={103881},
  year={2024},
  publisher={Elsevier}
}
```
