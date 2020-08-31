from __future__ import print_function, division

import os
import argparse

# import copy
# from model_summary import summary
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import csv
# pip install torchsummary

# Datasets

parser = argparse.ArgumentParser(description='Vision model evaluation')
# raw data paths
parser.add_argument('-d', '--data_path', type=str, dest='data_path', required=False, action='store',
                    default='../data/',
                    help='specify which data path')
parser.add_argument('-e', '--experiments_path', type=str, dest='experiments_path', required=False, action='store',
                    default="./experiments/", help='specify the data folder')
parser.add_argument('-fp', '--face_path', type=str, dest='face_path', required=False, action='store',
                    default='cropped',  # croppedcnn
                    help='specify which data path')

parser.add_argument('-i', '--inputs', dest='inputs', type=str, default='all'
                    , help='specify if face images (true) or the whole pictures should be used')
parser.add_argument('-he', '--head', type=str, dest='head', required=False, action='store',
                    default="t_complex", help='specify if the top layers of the model')
parser.add_argument('-m', '--model_name', type=str, dest='model_name', required=False, action='store',
                    default='all',
                    help='specify model name.')
                    # 'ResNet50', 'VGG-16', 'VGG-19', 'Xception', 'InceptionResNetV2', 'InceptionV3'
parser.add_argument('-c', '--cropped_faces', dest='cropped_faces', required=False, action='store_true'
                    , help='specify if face images (true) or the whole pictures should be used')
parser.add_argument('-t', '--testing_switch', dest='testing_switch', required=False,
                    action='store_true', help='specify if model should be run with few data')
parser.add_argument('-base', '--base_trainable', dest='base_trainable', required=False, type=bool,
                    default=True, help='specify if the base model should be trainable')

parser.add_argument('-ld', '--lr_decay_batch', dest='lr_decay_batch', required=False, action='store_true'
                    , help='specify if learning rate decay during epochs')
parser.add_argument('-free', '--freeze_unfreeze', dest='freeze_unfreeze', required=False, type=str, default='unfreeze'
                    , help='specify what type of freezing')
parser.add_argument('-pre', '--pretrained', dest='pretrained', required=False, type=str, default='imagenet'
                    , help='specify what weights are loaded')
parser.add_argument('-val', '--val_measure', dest='val_measure', required=False, type=str, default='val_acc'
                    , help='specify what measure should be tracked')
parser.add_argument('--data_augmentation', type=str, dest='data_augmentation', required=False, action='store',
                    default=None,
                    help='specify if data augementation on the trainset is used')

parser.add_argument('-p', '--partition', type=str, dest='partition', required=False, action='store',
                    default="devel", help='specify if the top layers of the model')
parser.add_argument('-sf', '--state_file', dest='state_file', required=False,
                    action='store_true', help='specify if model is a state file')

parser.add_argument('-g', '--gpu', dest='gpu', required=False, type=str,
                    default='0', help='specify if the gpu')
args = parser.parse_args()

if args.gpu is not None:
    print("Set GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from pathlib import Path
# import shutil
# from trainer import train_model
# from utils import class_weights, EarlyStopping,

# Own modules
from configs import prepare_config_and_experiments
from utils import export_predictions
from data import DatasetAll
from network.model import create_model
from metrics import acc, f1, plot_confusion_matrix  # evaluate_model
from network.model_summary import summary

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

print('-' * 25)
print(device)
print('-' * 25)
# Parameters

BATCH_SIZE = 10
TEST = False

if args.testing_switch:
    TEST = True
if TEST:
    BATCH_SIZE = 16  # 32 --> 12 steps
    DECAY_STEPS = 3
if args.model_name == 'NASNetLarge':
    print("reduce bs to avoid memory overflow")
    BATCH_SIZE = 16  # memory overflow

params = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'num_workers': 6}

if __name__ == "__main__":

    # Parameters
    config = prepare_config_and_experiments(args)

    config['best_model_dir'] = os.path.join(args.experiments_path, 'best_models', 'final')
    best_model_name = os.path.join(config['best_model_dir'], config['name'])  # +'.h5'

    if not os.path.exists(config['best_model_dir']):
        os.makedirs(config['best_model_dir'])

    labels = {}
    # Generators
    data_set = DatasetAll(config, partition='train')
    labels[args.partition] = data_set.get_labels()
    data_generator = torch.utils.data.DataLoader(data_set, **params)

    if args.partition == 'test':
        test_filenames = data_set.get_filenames()

    dataloaders = {args.partition: data_generator}

    if args.state_file:
        path_final = best_model_name + '.pt'
        model, input_size = create_model(config['feature_types'], args.inputs, args.head, args.pretrained)
        model.load_state_dict(torch.load(path_final, map_location=torch.device(device)))
        torch.save(model, best_model_name + '_state' + '.pt')
    else:
        path_final = best_model_name + '_state' + '.pt'
        model = torch.load(path_final, map_location=torch.device(device))

    model = model.to(device)
    model.eval()
    result, total_params, trainable_params = summary(model, input_size)
    print(result)

    prediction = {}

    print("Predict: {}".format(args.partition))

    train_per_epoch = len(data_set) // BATCH_SIZE
    if args.partition in ['train', 'val']:
        for i, (inputs, _) in enumerate(dataloaders[args.partition]):
            print("Progress {:2.1%} ".format(i / train_per_epoch), end="\r")
            inputs = [i.float().to(device) for i in inputs]
            outputs = model(*inputs)
            _, y_pred = torch.max(outputs, 1)
            prediction.setdefault(args.partition, []).append(y_pred.cpu().tolist())
    else:
        for i, (inputs) in enumerate(dataloaders[args.partition]):
            print("Progress {:2.1%} ".format(i / train_per_epoch), end="\r")
            inputs = [i.float().to(device) for i in inputs]
            outputs = model(*inputs)
            _, y_pred = torch.max(outputs, 1)
            prediction.setdefault(args.partition, []).append(y_pred.cpu().tolist())

    if args.partition in ['train', 'val']:
        # show results  
        _acc = acc(labels[args.partition], [item for sublist in prediction[args.partition] for item in sublist])
        _f1 = f1(labels[args.partition], [item for sublist in prediction[args.partition] for item in sublist])
        print('{} - acc: {}'.format(args.partition, _acc))
        print('{} - f1: {}'.format(args.partition, _f1))
        print('confusion matrix for {} exported.'.format(args.partition))
        plot_confusion_matrix(labels[args.partition],
                              [item for sublist in prediction[args.partition] for item in sublist]
                              , normalize=True, ticklabels=[str(i) for i in range(1, 9)]
                              , title='Confusion matrix {}'.format(args.partition), path=config['log_dir'])
    else:
        export_predictions(prediction[args.partition], test_filenames, data_set, name)
