# Music Source Separation

This repository contains the implementation for Homework 2 of the CommE5070 Deep Learning for Music Analysis and Generation course, Fall 2024, at National Taiwan University. For a detailed report, please refer to this [slides](https://docs.google.com/presentation/d/17R2Lj_37S_nfkOQE_zHsyDJTN8SrtzkPkNXv0no6mHE/edit?usp=sharing).


## Setup
To set up the virtual environment and install the required packages, use the following commands:
```
virtualenv --python=python3.10 deepmir_hw2
source deepmir_hw2/bin/activate
pip install -r requirements.txt
```

```
sudo apt install ffmpeg
```


## Data and Checkpoint Download

### Dataset
To download the dataset, run the following script:
```
bash scripts/download_data.sh
```

### Checkpoint
To download the pre-trained model checkpoints, use the command:
```
bash scripts/download_ckpt.sh
```


## Training
To train the model, run the command:
```
bash scripts/train.sh
```

## Inference
To inference the model, run the command:
```
bash scripts/inference.sh
```


## Environment
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
If you use this code, please cite the following:
```bibtex
@misc{source_separation_2024,
    title  = {Source Separation},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Source-Separation},
    year   = {2024}
}
```
```bibtex
@article{stoter2019open,
    title   = {Open-unmix-a reference implementation for music source separation},
    author  = {St{\"o}ter, Fabian-Robert and Uhlich, Stefan and Liutkus, Antoine and Mitsufuji, Yuki},
    journal = {Journal of Open Source Software},
    year    = {2019}
}
```
