# HACNet: End-to-end learning of tabular-to-image converter and convolutional neural network

This repository contains the code for the following paper:

Preprint:https://www.researchgate.net/publication/364456841_HACNet_End-to-end_learning_of_table-to-image_converter_and_convolutional_neural_network

Please cite our paper if you use this code for your research.

## Directory
.  
├── README.md  
├── src  (source code of HACNet)  
├── datasets  (Please put datasets here)  
└── template  (Template images we used)  

## Requirements
You can utilize [Pipfile](https://github.com/shiralab/table2image/blob/main/Pipfile) or [requirements.txt](https://github.com/shiralab/table2image/blob/main/requirements.txt) to set up the environment.

We checked the code works in the following environment:
- Ubuntu 18.04 LTS
- GPU: NVIDIA A100
- CUDA version 11.5
- Python 3.9
- PyTorch 1.9.0

## Usage
You can download datasets from [these links](https://github.com/shiralab/table2image/blob/main/datasets/link.md). Please preprocess downloaded data and put correctly to read ([ref](https://github.com/shiralab/table2image/blob/main/src/manager/data_manager.py#L32)).

You can determine the setting from [config.yaml](https://github.com/shiralab/table2image/blob/main/src/conf/config.yaml) file. 

Then, run the program as
```sh
$ cd src
$ python main.py
```
