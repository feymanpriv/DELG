# DELG-pytorch
Pytorch Implementation of Unifying Deep Local and Global Features for Image Search

- DELG pipline:
<p align="center"><img width="90%" src="extraction/vis/delg_pipline.png" /></p>

## Installation

Install Python dependencies:

```
pip install -r requirements.txt
```

Set PYTHONPATH:

```
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Training

Training a delg model:

```
python train_delg.py \
    --cfg configs/metric/resnet_delg_8gpu.yaml \
    OUT_DIR ./output \
    PORT 12001 \
    TRAIN.WEIGHTS path/to/pretrainedmodel
```

## Feature extraction

Extracting global and local feature for multi-scales
```
python extraction/extractor.py --cfg configs/resnet_delg_8gpu.yaml
```
Refer [`extractor.sh`](extraction/extract.sh) for using multicards

See [`visualize.ipynb`](extraction/vis/attention/visualize.ipynb) for verification of local features

## Evaluation on ROxf and RPar

See (https://github.com/filipradenovic/revisitop) for details

### Local Match

- Spatial Verification
```
cd extraction/revisitop
python example_evaluate_with_local.py main
```

- ASMK (updating)

- Examples
<p align="center"><img width="90%" src="extraction/vis/matches/match_example_1.jpg" /></p>

### Results

(on going)
