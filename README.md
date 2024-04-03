conda create --name behaviour python=3.9 -y
conda activate behaviour
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .


python build_labels.py -r ./data -v videos -a ./animal-kingdom/AR_metadata.xlsx

python build_labels.py -r ./data -v rawframes -a ./animal-kingdom/AR_metadata.xlsx -ra

python mmaction2/tools/train.py work_dirs/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py


docker build -f ./docker/Dockerfile --rm -t mmaction2 .

docker run --gpus all --shm-size=24g -it -v /home/sadat/Desktop/animal_behaviour/data:/mmaction2/data mmaction2



