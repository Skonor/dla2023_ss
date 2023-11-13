
## Overview

This repo contains framework for SS training with Spex+ (classification head included)



## Installation guide

```shell
pip install -r ./requirements.txt
```

To load checkpoints run:
```shell
python scripts/load_checkpoints.py
```

## Dataset creation:

1. Install train-clean-100 and test-clean parts from librispeech

```shell
python scripts/load_librispeech.py
```

2. Creaete 10k (100 speakers) mixes for training and 1k mixes for evaluation

```shell
python scripts/create_mixes.py
```
This will create train and val directories in directory in data/datasets/LibriMixes, each containig 3 directories mix, refs, targets with audio files.

## Training

To reproduce training do the following (All training was done on kaggle, so you will need to change paths in config)

1. Train for 20k steps (batch size = 3)

```shell
python train.py -c hw_asr/configs/SpexPlus_config/config.json
```

2. Finetune model for 20k steps with the same config and dataset:
```
python train.py -c hw_asr/configs/SpexPlus_config/config.json -f saved/checkpoints/spex_plus_20k/spex_plus_20k/model_weights.pth
```

(Here I used checkpoint loaded by load_checkpoints.py script. You can change -f path to your local weights destination)

## Evaluation

For evaluating models on custom dataset do the following:

1. Load checkpoints from training:
```shell
python scripts/load_checkpoints.py
```
This will create checkpoints dirs is saved/models/ contaning model weigths file and training config.

2. Run test.py:
```shell
python test.py /librispeech_clean.json -r saved/checkpoints/spex_plus_finetuned/model_weights.pth -t <your_directory>
```

This will print out SI-SDR and CER metrics for each of prediction methods


## Results

For Spex+ model trained for 20k steps we get the following results:

| data   | SI-SDR        | PESQ           | 
|--------|---------------|----------------|
| eval   |  8.24         |     1.92       | 
| public test |  7.35    |     1.38       |   

And after training for another 20k strps:

| data   | SI-SDR        | PESQ           | 
|--------|---------------|----------------|
| eval   |  9.70         |     2.07       | 
| public test |  8.83    |     1.54       |   

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
