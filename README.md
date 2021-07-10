# Experiment Weakly Segmentation

## Requirements
- Linux (tested on Ubuntu 20.04 using WSL2)
- conda installed

## Download Data
- be sure you have **zip** command available 
    - `sudo apt install zip`

## Create Conda Env
- `conda env create -f environment.yml || conda env update -f environment.yml`

## Recreate Conda Env from scratch
- `conda create -n segweak python=3.7 -y`
- `conda activate segweak`

- `conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y`
- `conda install -c conda-forge opencv -y`
- `conda install scikit-learn -y`

- `pip install git+https://github.com/qubvel/segmentation_models.pytorch --quiet`
- `pip install -U git+https://github.com/albu/albumentations --no-cache-dir --quiet`


- `conda env export --no-builds | grep -v "^-e" | grep -v "prefix" > environment.yml`