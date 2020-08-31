# Accompanying code to the paper: X-AWARE: ConteXt-AWARE Human-Environment Attention Fusion for Driver Gaze Prediction in the Wild

*Reliable systems for automatic estimation of the driver's gaze are crucial for reducing the number of traffic fatalities and for many emerging research areas aimed at developing intelligent vehicle-passenger systems. Gaze estimation is a challenging task, especially in environments with varying illumination and reflection properties. Furthermore, there is wide diversity with respect to the appearance of drivers' faces, both in terms of occlusions (e.g. vision aids) and cultural/ethnic backgrounds. For this reason, analysing the face along with contextual information -- for example, the vehicle cabin environment -- adds another, less subjective signal towards the design of robust systems for passenger gaze estimation.
In this paper, we present an integrated approach to jointly model different features for this task. In particular, to improve the fusion of the visually captured environment with the driver's face, we have developed a contextual attention mechanism, X-AWARE, attached directly to the output convolutional layers of InceptionResNetV networks. In order to showcase the effectiveness of our approach, we use the Driver Gaze in the Wild dataset, recently released as part of the Eighth Emotion Recognition in the Wild Challenge (EmotiW) challenge. Our best model outperforms the baseline by an absolute of 15.43% in accuracy on the validation set, and improves the previously best reported result by an absolute of 8.72% on the test set.*

## Installation
```
pip install -r requirements.txt
```

## Run
```
usage: end2end.py [-h] [-d DATA_PATH] [-fp FACE_PATH] [-e EXPERIMENTS_PATH]
                  [-i INPUTS] [-he HEAD] [-m MODEL_NAME] [-ld]
                  [-free FREEZE_UNFREEZE] [-pre PRETRAINED] [-val VAL_MEASURE]
                  [-base BASE_TRAINABLE]
                  [--data_augmentation DATA_AUGMENTATION] [-t] [-g GPU]

Vision model evaluation

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
                        specify which data path
  -fp FACE_PATH, --face_path FACE_PATH
                        specify which data path
  -e EXPERIMENTS_PATH, --experiments_path EXPERIMENTS_PATH
                        specify the data folder
  -i INPUTS, --inputs INPUTS
                        specify if face images (true) or the whole pictures
                        should be used
  -he HEAD, --head HEAD
                        specify if the top layers of the model
  -m MODEL_NAME, --model_name MODEL_NAME
                        specify model name.
  -ld, --lr_decay_batch
                        specify if learning rate decay during epochs
  -free FREEZE_UNFREEZE, --freeze_unfreeze FREEZE_UNFREEZE
                        specify what type of freezing
  -pre PRETRAINED, --pretrained PRETRAINED
                        specify what weights are loaded
  -val VAL_MEASURE, --val_measure VAL_MEASURE
                        specify what measure should be tracked
  -base BASE_TRAINABLE, --base_trainable BASE_TRAINABLE
                        specify if the base model should be trainable
  --data_augmentation DATA_AUGMENTATION
                        specify if data augementation on the trainset is used
  -t, --testing_switch  specify if model should be run with few data
  -g GPU, --gpu GPU     specify if the gpu
```
Further settings in configs.py.

## Run best model
1. Download best model: https://drive.google.com/file/d/1cMly7J12YWpTWzz5K2inp9fYLg1n1-1g/view?usp=sharing
2. Copy to XAWARE/x_aware/experiments/best_models/final
2. Run
```
python predict_v1.py
```

## Acknowledgement
Some parts are adjusted and extend from other git. Check out: 
- model_summary.py: Derived from https://github.com/sksq96/pytorch-summary
- stand_alone_blocks.py: Stand-Alone-Self-Attention https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py
- inceptionresnet.py: https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py
All credits to the authors.