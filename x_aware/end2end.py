from __future__ import print_function, division

import csv
import os
import argparse

# Arguments
parser = argparse.ArgumentParser(description='Vision model evaluation')
# raw data paths
parser.add_argument('-d', '--data_path', type=str, dest='data_path', required=False, action='store',
                    default='../data/',
                    help='specify data path')
parser.add_argument('-fp', '--face_path', type=str, dest='face_path', required=False, action='store',
                    default='cropped',  # croppedcnn
                    help='specify face img path')
parser.add_argument('-e', '--experiments_path', type=str, dest='experiments_path', required=False, action='store',
                    default="./experiments/", help='specify the data folder')
# network parameters
parser.add_argument('-i', '--inputs', dest='inputs', type=str, default='all'
                    , help='specify if face images (true) or the whole pictures should be used [env_i/face_i/all_i/face_f/all]')
parser.add_argument('-he', '--head', type=str, dest='head', required=False, action='store',
                    default="t_complex", help='specify if the top layers of the model e.g. pool, direct, etc.')
parser.add_argument('-m', '--model_name', type=str, dest='model_name', required=False, action='store',
                    default='all', help='specify model name. [deactivated]')
#  todo: integrate 'ResNet50', 'VGG-16', 'VGG-19', 'Xception', 'InceptionResNetV2', 'InceptionV3' including weights here
parser.add_argument('-ld', '--lr_decay_batch', dest='lr_decay_batch', required=False, action='store_true'
                    , help='specify if learning rate decay during epochs')
parser.add_argument('-free', '--freeze_unfreeze', dest='freeze_unfreeze', required=False, type=str, default='unfreeze'
                    , help='specify type of freezing')
parser.add_argument('-pre', '--pretrained', dest='pretrained', required=False, type=str, default='imagenet'
                    , help='specify if pretrained weights are loaded')
parser.add_argument('-val', '--val_measure', dest='val_measure', required=False, type=str, default='val_acc'
                    , help='specify measure to track')
parser.add_argument('-base', '--base_trainable', dest='base_trainable', required=False, type=bool,
                    default=True, help='specify if the base model should be trainable')
parser.add_argument('--data_augmentation', type=str, dest='data_augmentation', required=False, action='store',
                    default=None,
                    help='specify if data augementation on the trainset is used [deactivated]')  # todo: add usage
# envr. settings
parser.add_argument('-t', '--testing_switch', dest='testing_switch', required=False,
                    action='store_true', help='specify if model should be run with few data')
parser.add_argument('-g', '--gpu', dest='gpu', required=False, type=str,
                    default=None, help='specify if and which gpu')
args = parser.parse_args()

# import torch after setting gpu
if args.gpu is not None:
    if args.gpu == 'cpu':
        print("Set GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# Own modules
from configs import prepare_config_and_experiments
from utils import class_weights, EarlyStopping
from data import DatasetAll
from network.model import create_model
from trainer import train_model
from metrics import output_graphs, evaluate_model
from network.model_summary import summary

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
if args.gpu == 'cpu':
    device = torch.device("cpu")    
else:
    device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print('-' * 25)
print(args.gpu)
print(device)
print('-' * 25)

if __name__ == "__main__":

    # parameters
    config = prepare_config_and_experiments(args)

    # generators
    training_set = DatasetAll(config, partition='train')
    training_generator = torch.utils.data.DataLoader(training_set, **config['params_train'])

    validation_set = DatasetAll(config, partition='val')
    validation_generator = torch.utils.data.DataLoader(validation_set, **config['params_val'])

    test_set = DatasetAll(config, partition='test')
    test_generator = torch.utils.data.DataLoader(test_set, **config['params_test'])

    dataloaders = {'train': training_generator, 'val': validation_generator}
    dataset_sizes = {'train': len(training_set), 'val': len(validation_set)}

    # model
    model_ft, input_size = create_model(config['feature_types'], args.inputs, args.head, args.pretrained)
    model_ft = model_ft.to(device)
    result, total_params, trainable_params = summary(model_ft, input_size)
    weights = class_weights(training_set.get_labels())
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=config['LEARNING_RATE'])

    # decay LR by a factor of
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft,
                                                  gamma=0.95)  # StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(device, config
                                   , monitor=config['TRACKING_MEASURE']
                                   , test_set=test_set
                                   , test_generator=test_generator
                                   , verbose=True)

    train_per_epoch = dataset_sizes['train'] // config['BATCH_SIZE']

    model, history = train_model(device
                                 , config
                                 , model_ft
                                 , dataloaders
                                 , dataset_sizes
                                 , criterion
                                 , optimizer_ft
                                 , exp_lr_scheduler
                                 , name=config['name']
                                 , early_stopping=early_stopping
                                 , train_per_epoch=train_per_epoch)

    # export experiment information
    output_graphs(config['log_dir'], history)

    with open(os.path.join(config['log_dir'], 'overall.csv'), 'w+', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in history.items():
            maxvalue = max(value)
            writer.writerow([key, value, maxvalue])

    with open(os.path.join(config['log_dir'], 'modelsummary.txt'), 'w') as f:
        f.write(result)
        f.write("total_params: %s" % total_params)
        f.write("trainable_params: %s" % trainable_params)

    with open(os.path.join(config['log_dir'], 'config.txt'), 'w') as f:
        for k, v in config.items():
            f.write(str(k) + ',' + str(v))

    evaluate_model(model, training_generator, dataset_sizes, device, 'train', config)
    evaluate_model(model, validation_generator, dataset_sizes, device, 'val', config)