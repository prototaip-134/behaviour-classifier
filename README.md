# behaviour-classifier

This repo contains the setup for our training and evaluation of models for  behaviour classification using [MMAction2](https://github.com/open-mmlab/mmaction2). Currently, we support the [Animal-Kingdom](https://sutdcv.github.io/Animal-Kingdom/) dataset but more datasets will be added later.

Clone recursively for MMaction2:
```
git clone --recurse https://github.com/sttaseen/scrambmix.git
```

<div align="center">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/prototaip-134/animal-behaviour/assets/67076071/0ee24bbd-dac9-4e07-bb2c-362622254acb"
  height=auto
  ><br>
    <p style="font-size:1.5vw;">"Dancing" from Animal-Kingdom Dataset</p>
  </div>
</div>

## Directory
**After** downloading and extracting the dataset, the directory for a `RawframeDataset` should look like below:
```
root-repo
├── data
│   └── dataset-name
│       ├── rawframes
│       │   ├── test
│       │   │   └── 00623
│       │   │       ├── img_00001.jpg
│       │   │       └── ...
│       │   ├── train
│       │   │   └── 00623
│       │   │       ├── img_00001.jpg
│       │   │       └── ...
│       │   └── val
│       │       └── 00626
│       │           ├── img_00001.jpg
│       │           └── ...
│       ├── test_annotations.txt
│       ├── train_annotations.txt
│       ├── val_annotations.txt
│       └── dataset-name.zip
├── mmaction2
│   └── ...
├── README.md
├── tools
│   └── dataset-name
│       └── build_labels.py
└── work_dirs
    └── ...
```

In-depth information about how to set up each dataset can be found in the ```README.md``` in their respective ```tools/<dataset-name>``` folder. `build_labels.py` can be used to create annotations for both `RawframeDataset` and `VideoDataset`. For custom datasets, MMAction2 can be used to set up the datasets. More information can be found [here](https://mmaction2.readthedocs.io/en/latest/user_guides/prepare_dataset.html).

## Requirements
### Using Docker
The docker file for Ubuntu 20.04 with Torch 2.2.2 and CUDA 11.8.0 can be found [here](docker/Dockerfile). This image contains all the dependencies needed to run MMAction2. You can pull the image from dockerhub by running the following:
```
docker pull sttaseen/mmaction2:latest
```
Then run the following to run a container:
```
docker run -v <Absolute path to the dataset folder>:/data -it --shm-size=32g --gpus all  sttaseen/mmaction2
```

### Setting up a conda environment (Optional)
#### Install MiniConda
If you are using the docker image, then you can ignore these step. The following instructions are for Bash. For other operating systems, download and install from [here](https://docs.conda.io/en/latest/miniconda.html).
```
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
 "Miniconda3.sh"
```
Install the .sh file.
```
bash Miniconda3.sh
```
Remove the installer:
```
rm Miniconda3.sh
```
#### Creating a virtual environment
Run the following commands to create a virtual environment and to activate it:
```
conda create -n behaviour python=3.8 -y
conda activate behaviour
```
Make sure to run ```conda activate behaviour``` before running any of the scripts in this repo.

### Installing Dependencies
For non-Mac OS, install PyTorch by running the following:
```
conda install pytorch torchvision -c pytorch
```
If you are on Mac OS with MPS acceleration, run the following instead:
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

**Note:** To fully utilise cuda, make sure to have nvidia graphics drivers installed and running. To check, run ```nvidia-smi```.


To install all the other modules, navigate to the root directory of this repo after cloning and run the following:
```
pip install -r requirements.txt
```
Install [Decord](https://github.com/dmlc/decord). For Mac OS, check their [repo](https://github.com/dmlc/decord#mac-os) for more info.
```
pip install decord==0.6.0
```
Install mmcv:
```
pip install -U openmim
mim install mmcv-full
```
For non-Mac OS, assuming the current directory is the root of the repository, install mmaction2 from source:
```
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .  
cd ..
```

If you are on Mac OS, run the following instead:
```
cd mmaction2
pip install -r requirements/build.txt
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e . 
cd ..
```

To set up our custom data augments, head over to `experiments/scrambmix` and run `copy_blend.sh`. This will copy the scrambmix augment over to MMAction2:
```
cd experiments/scrambmix
bash copy_blend.sh
```

This one is optional but to use the conda environment in Notebook, run:
```
conda install ipykernel -y
ipython kernel install --user --name=behaviour
```

## Setting up WandB (Optional)
Add the WandB hook by replacing `vis_backends`:
```
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend')
]
visualizer = dict(
    type='ActionVisualizer', 
    vis_backends=vis_backends
)
```

## Training
Refer to the official MMAction2 [documentation](https://mmaction2.readthedocs.io/en/latest/user_guides/train_test.html) for specific details.
```
python mmaction2/tools/train.py ${CONFIG_FILE} [optional arguments]
```
Example: Train a pretrained 3CD model on wlasl with periodic validation.
```
python mmaction2/tools/train.py models/c3d/c3d_16x16x1_sports1m_wlasl100_rgb.py --validate --seed 0 --deterministic --gpus 1
```

## Testing
Use the following template to test a model:
```
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```
Examples and more information can be found [here](https://github.com/open-mmlab/mmaction2/blob/master/docs/en/getting_started.md#test-a-dataset).


## Citations
```
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
```



