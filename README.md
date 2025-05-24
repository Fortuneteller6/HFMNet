<div id="top" align="center">

### A Lightweight Feature Enhancement Model for Infrared Small Target Detection

Kuanhong Cheng, Teng Ma, Rong Fei, and Junhuai Li.

<hr/>

[![](https://img.shields.io/badge/ISJ-2025.3549519-green.svg?style=flat-square)]([10.1109/JSEN.2025.3549519](https://doi.org/10.1109/JSEN.2025.3549519))
![](https://img.shields.io/badge/Language-Python-blue.svg?style=flat-square)

</div>

### Datasets Prepare

- IRSTD-1K dataset is available at [IRSTD-1K](https://github.com/RuiZhang97/ISNet).
- NUAA-SIRST dataset is available at [NUAA-SIRST](https://github.com/YimianDai/sirst).
- NUDT-SIRST dataset is available at [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection).
- DenseSIRST dataset is available at [DenseSIRST](https://github.com/GrokCV/DenseSIRST).
- We also prepare the txt file for dividing dataset and three datasets, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1bCbrS5B2BWyUjK2Ic0nyreu4wZ9omgpY?usp=sharing).

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
| HFMNet |  IRSTD-1K  | 92.52 | 4.18  | 70.38 |  82.62   | [Weights](https://drive.google.com/drive/folders/164Y9BWa9alPE41YZzw5HX7JT8aaiCKkE?usp=sharing) |
| HFMNet | NUAA-SIRST | 99.08 | 0.35  | 79.16 |  88.37   | [Weights](https://drive.google.com/drive/folders/12TTrVDannF9KnpOfwbse5frrdoGgx6gQ?usp=sharing) |
| HFMNet | NUDT-SIRST | 99.30 | 6.05  | 91.34 |  95.47   | [Weights](https://drive.google.com/drive/folders/18lm0Jdc33nCzPZwIHJw4CkWupj9qU3lf?usp=sharing) |
| HFMNet | DenseSIRST | 91.34 | 8.52 | 69.35 |  81.90   | [Weights](https://drive.google.com/drive/folders/1mVmbYtUyrGxPRAHLOOOMIdXgL24QlqLN?usp=sharing) |

<hr/>

### Acknowledgement

The code of this paper is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks for their awesome work.

### Citation

If you find the code helpful in your resarch or work, please cite this paper as following.

```
@article{HFMNet,
  author={Cheng, Kuanhong and Ma, Teng and Fei, Rong and Li, Junhuai},
  journal={IEEE Sensors Journal}, 
  title={A Lightweight Feature Enhancement Model for Infrared Small Target Detection}, 
  year={2025},
  volume={25},
  number={9},
  pages={15224-15234}
}
```

If the above article has reference value for your work, our team's other IRSTD works can also serve as references. [MDCENet](https://www.sciencedirect.com/science/article/abs/pii/S1350449524003591) | [WaveTD](https://www.sciencedirect.com/science/article/abs/pii/S1350449525001434)

```
@article{MDCENet,
  title={Mdcenet: Multi-dimensional cross-enhanced network for infrared small target detection},
  author={Ma, Teng and Cheng, Kuanhong and Chai, Tingting and Prasad, Shitala and Zhao, Dong and Li, Junhuai and Zhou, Huixin},
  journal={Infrared Physics \& Technology},
  volume={141},
  pages={105475},
  year={2024},
  publisher={Elsevier}
}

@article{WaveTD,
  title={An Wavelet Steered network for efficient infrared small target detection},
  author={Ma, Teng and Cheng, Kuanhong and Chai, Tingting and Wu, Yubo and Zhou, Huixin},
  journal={Infrared Physics \& Technology},
  volume = {148},
  pages={105850},
  year={2025},
  publisher={Elsevier}
}
```

### Contact

If you have any questions, please feel free to reach me out at teng_m@yeah.net
