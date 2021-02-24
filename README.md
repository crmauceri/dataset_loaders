# pytorch-dataloaders

### Introduction
- [x] Load RGB-D datasets Cityscapes, COCO, SUNRGBD, and SceneNetRGBD
- [x] Optional RGB-D network input using 4th channel in first convolutional layer 
- [x] YACS configuration files

Based on data loading code from [jfzhang95/pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception.git)

### Installation
The code was tested with Anaconda and Python 3.8. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/crmauceri/dataset_loaders.git
    cd dataset_loaders
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    Other dependencies:
    ```Shell
    conda install matplotlib pillow tqdm protobuf scipy numpy
    pip install yacs
    ```
   Coco tools
   ```bash
   conda install -c conda-forge pycocotools scikit-image
   ```
    
2. Compile SceneNetRGBD protobuf files
   ```bash
   cd dataloaders/datasets
   make
   ```  

3. Install as module:
   ```bash
   cd $root
   pip install -e .
   ```

### Visualization

Jupyter notebook `COCO_Data_Browser.ipynb` is provided for visualizing data and trained network results. 