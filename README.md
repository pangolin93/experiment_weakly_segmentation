# Experiment Weakly Segmentation

## Intro
I want to perform and test various weakly learning techniques. 
I will experiment a setup for a semantic segmantation using classification data as weak labels. 

Weak labels are created at tile-level, in particular classification labels are using the 5 classes.
    - for each class I compute the percentage of pixels 
    - those percentages are the new weak labels (e.g. [ 0.5, 0.1, 0.3, 0.1, 0]).

## Data
I will use Data from ISPRS (https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/).
So I will divide all data in 3 sets
- 10% train 
    - where strong labels are available
- 70% weak
    - I can use only weak labels 
- 20% validation 

## Idea

I structure data in this way.
- I will train a FPN model with just strong labels
- I measure performances on validation data

- Then I want to exploit weak labels also. 
    - The idea is to "pre-train" Segmentation models only with weak labels.
    - Check performances on validation
    - Then i will fine tune that model with only strong labels
    - and finally i will see metrics again on validation to see if there is an improvement or not


Weak training steps:
- Take a segmantation model (where the encoder part is pretrained on imagenet)
- Pass the input data (the satellite image) to the model
- Take predicted output and compute the percentage of pixels in frame for each class, creating the prediction for weak label
- Compare weak prediction with weak label using loss and metrics from classification problem

## Requirements
- Linux (tested on Ubuntu 20.04 using WSL2)
- conda installed
- GPU
    - NVIDIA GeForce RTX 3060 Laptop GPU
    - NVIDIA-SMI 470.28
    - Driver Version: 470.76
    - CUDA Version: 11.4

## Download Data
- you will use Data from ISPRS (https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)
    - be sure you have **zip** command available 
        - `sudo apt install zip`
- to download data
    - check you are in the root folder
    - execute `bash download_data.sh <your_username> <your_pwd>`
## Create Conda Env
- `conda env create -f environment.yml || conda env update -f environment.yml`

## Recreate Conda Env from scratch
- `conda create -n segweak python=3.7 -y`
- `conda activate segweak`

- `conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y`
- `conda install -c conda-forge opencv -y`
- `conda install scikit-learn -y`
- `conda install pandas matplotlib jupyter -y`

- `pip install git+https://github.com/qubvel/segmentation_models.pytorch --quiet`
- `pip install -U git+https://github.com/albu/albumentations --no-cache-dir --quiet`


- `conda env export --no-builds | grep -v "^-e" | grep -v "prefix" > environment.yml`




# References
- Several lines of code are inspired by this notebook!
    - https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
