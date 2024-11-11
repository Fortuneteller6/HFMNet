### 📖 A Lightweight Feature Enhancement Model for Infrared Small Target Detection

<hr/>

[![](https://img.shields.io/badge/Building-Done-green.svg?style=flat-square)](https://github.com/Fortuneteller6/HFMNet) ![](https://img.shields.io/badge/Language-Python-blue.svg?style=flat-square) [![](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](./LICENSE)

> [Paper Link]()  
> Authors: Kuanhong Cheng, Teng Ma, Rong Fei, and Junhuai Li. <br/>
> The code and model weights will be made public after the paper is accepted. Thanks for your attention!

<hr/>

### Datasets Prepare

- IRSTD-1K dataset is available at [IRSTD-1K](https://github.com/RuiZhang97/ISNet).
- NUAA-SIRST dataset is available at [NUAA-SIRST](https://github.com/YimianDai/sirst).
- NUDT-SIRST dataset is available at [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection).
- DenseSIRST dataset is available at [DenseSIRST](https://github.com/GrokCV/DenseSIRST).
- We also prepare the txt file for dividing dataset and three datasets, which can be downloaded from [Google Drive]().

<hr/>

### Commands for Taining

- The epoch and bath size for training the [HFMNet](https://github.com/Fortuneteller6/HFMNet) can be found in the following commands.

```python
python train.py --base_size 256 --crop_size 256 --epochs 500 --dataset IRSTD-1K --split_method 80_20 --model HFMNet --deep_supervision True --train_batch_size 4 --test_batch_size 4 --mode TXT
```

```python
python train.py --base_size 256 --crop_size 256 --epochs 1500 --dataset NUAA-SIRST --split_method 80_20 --model HFMNet --deep_supervision True --train_batch_size 4 --test_batch_size 4 --mode TXT
```

```python
python train.py --base_size 256 --crop_size 256 --epochs 1500 --dataset NUDT-SIRST --split_method 80_20 --model HFMNet --deep_supervision True --train_batch_size 8 --test_batch_size 8 --mode TXT
```

```python
python train.py --base_size 256 --crop_size 256 --epochs 500 --dataset DenseSIRST --split_method 80_20 --model HFMNet --deep_supervision True --train_batch_size 8 --test_batch_size 8 --mode TXT
```

### Commands for Testing and Visulization

- For both testing and visulization of different dataset, you just need to change the model weights and the dataset name.

```python
python test.py --base_size 256 --crop_size 256 --st_model IRSTD-1K_HFMNet_11_10_2024_07_41_13_wDS --model_dir IRSTD-1K_HFMNet_11_10_2024_07_41_13_wDS/mIoU__HFMNet_IRSTD-1K_epoch.pth.tar --dataset IRSTD-1K --split_method 80_20 --model HFMNet --deep_supervision True --test_batch_size 1 --mode TXT
```

```python
python visulization.py --base_size 256 --crop_size 256 --st_model IRSTD-1K_HFMNet_11_10_2024_07_41_13_wDS --model_dir IRSTD-1K_HFMNet_11_10_2024_07_41_13_wDS/mIoU__HFMNet_IRSTD-1K_epoch.pth.tar --dataset IRSTD-1K --split_method 80_20 --model HFMNet --deep_supervision True --test_batch_size 1 --mode TXT
```

<hr/>

### Results and Weights

|  Methods   |    Data    |  Pd   |  Fa   |  IoU  | F1_Score |  Download   |
| :--------: | :--------: | :---: | :---: | :---: | :------: | :---------: |
| HFMNet |  IRSTD-1K  | 92.52 | 4.18  | 70.38 |  82.62   | [Weights]() |
| HFMNet | NUAA-SIRST | 99.08 | 0.35  | 79.16 |  88.37   | [Weights]() |
| HFMNet | NUDT-SIRST | 99.30 | 6.05  | 91.34 |  95.47   | [Weights]() |
| HFMNet | DenseSIRST | 91.34 | 8.52 | 69.35 |  81.90   | [Weights]() |

<hr/>

### Acknowledgement

The code of this paper is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks for their awesome work.

### Citation

If you find the code helpful in your resarch or work, please cite this paper as following.

```

```

### Contact

If you have any questions, please feel free to reach me out at teng_m@yeah.net
