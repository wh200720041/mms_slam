## Installation
This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection)(v2.0.0)

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.3 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv>=0.5.1, <=0.5.8]

We have tested the following versions of OS and softwares:

- OS: Ubuntu 18.04
- CUDA: 10.2
- GCC 7.5
- NCCL 2.7.8

### Install SOLO

a. Create a conda virtual environment and activate it.

```shell
conda create -n name python=3.7 -y
conda activate name
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone this repository.

```shell
git clone xxx
cd xxx
```

d. Install build requirements and then install SOLO.
(We install pycocotools via the github repo instead of pypi because the pypi version is old and not compatible with the latest numpy.)

```shell
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .  # or "python setup.py develop"
```

### Prepare datasets

It is recommended to symlink the dataset root to `$SOLO/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
SOLO
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```
The cityscapes annotations have to be converted into the coco format using the [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) toolbox.
We plan to provide an easy to use conversion script. For the moment we recommend following the instructions provided in the
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark/data) toolbox. When using this script all images have to be moved into the same folder. On linux systems this can e.g. be done for the train images with:
```shell
cd data/cityscapes/
mv train/*/* train/
```

### A from-scratch setup script

Here is a full script for setting up SOLO with conda and link the dataset path (supposing that your COCO dataset path is $COCO_ROOT).

```shell
conda create -n name python=3.7 -y
conda activate name

conda install -c pytorch pytorch torchvision -y
conda install cython -y
git clone xxx
cd xxx
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .

mkdir data
ln -s $COCO_ROOT data
```

## Usage

### Train with GPUs
    python tools/train.py ${CONFIG_FILE} --gpus {}
    
    Example:
    python tools/train.py configs/solo/solo_r50_fpn_8gpu_1x.py --gpus 8

### Testing
    # multi-gpu testing
    ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  --show --out  ${OUTPUT_FILE} --eval segm
    
    Example: 
    ./tools/dist_test.sh configs/solo/solo_r50_fpn_8gpu_1x.py SOLO_R50_1x.pth  8  --show --out results_solo.pkl --eval segm

    # single-gpu testing
    python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm
    
    Example: 
    python tools/test_ins.py configs/solo/solo_r50_fpn_8gpu_1x.py  SOLO_R50_1x.pth --show --out  results_solo.pkl --eval segm


### Visualization

    python tools/test_ins_vis.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --save_dir  ${SAVE_DIR}
    
    Example: 
    python tools/test_ins_vis.py configs/solo/solo_r50_fpn_8gpu_1x.py  SOLO_R50_1x.pth --show --save_dir  work_dirs/vis_solo
