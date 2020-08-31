from __future__ import print_function, division

import os
import argparse
import torch

# Own
from data_v1 import DatasetAll
from metrics import acc, f1
from network.model import create_model
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description = 'Paper model v1 evaluation')
# raw data paths
parser.add_argument('-d', '--data_path', type = str, dest = 'data_path', required = False, action = 'store', 
                    default = '../data/', 
                    help = 'specify which data path')
parser.add_argument('-e', '--experiments_path', type = str, dest = 'experiments_path', required = False, action = 'store',
                    default = "./experiments/", help = 'specify the data folder')
parser.add_argument('-i', '--inputs', dest = 'inputs', type = str, default = 'all_i'
                    , help = 'specify if face images (true) or the whole pictures should be used')
parser.add_argument('-he','--head', type = str, dest = 'head', required = False, action = 'store',
                    default = "poolv1", help = 'specify if the top layers of the model')
parser.add_argument('-p','--partition', type = str, dest = 'partition', required = False, action = 'store',
                    default = "val", help = 'specify if the top layers of the model')
parser.add_argument('-t', '--testing_switch', dest = 'testing_switch', required = False, 
                    action = 'store_true', help = 'specify if model should be run with few data')
parser.add_argument('-g', '--gpu', dest = 'gpu', required = False, type = str,
                    default = '0', help = 'specify if the gpu')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
#torch.autograd.detect_anomaly(True)

print('-'*25)
print(device)
print('-'*25)
# Parameters

BATCH_SIZE = 10
TEST = False

if args.testing_switch:
    TEST = True
if TEST:
    BATCH_SIZE = 16 #32 --> 12 steps
    DECAY_STEPS = 3

params = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'num_workers': 6}

def export(test_prediction, test_filenames,data_set,name):
    export_path = os.path.join('../fumo/', name)

    dirpath = Path(export_path)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    test_flat = [item for sublist in test_prediction for item in sublist]
    test_flat = data_set.get_real_labels_fromListoLabels(test_flat)

    for i, file in enumerate(test_filenames):
        with open(os.path.join(export_path, file.split('.')[0] + '.txt'), "w") as file:
            file.write(str(test_flat[i]))

if __name__  == "__main__":

    model_name = 'InceptionResNetV2'
    feature_types = []

    if args.inputs == 'env_i' or args.inputs == 'all_i':
        feature_types.append('env_img_extractor')
    if args.inputs == 'face_i' or args.inputs == 'all_i':
        feature_types.append('faces_img_extractor')
    if args.inputs == 'face_f':
        feature_types.append('face_extractor')
    if args.inputs == 'all':
        feature_types = ['env_img_extractor','faces_img_extractor','face_extractor','gocar']

    name = 'env_img_faces_img_InceptionResNetV2_btrainable_poolv1'
    best_model_dir = os.path.join(args.experiments_path, 'best_models','final')
    best_model_name = os.path.join(best_model_dir, name)

    # Data
    data_set = DatasetAll(data_path = args.data_path
                        , partition=args.partition
                        , feature_types=feature_types
                        , testing =TEST)
    labels = {}
    labels[args.partition] = data_set.get_labels()      
    data_generator = torch.utils.data.DataLoader(data_set, **params)

    if args.partition == 'test':
        test_filenames = data_set.get_filenames()

    dataloaders = {args.partition: data_generator}

    path_final = best_model_name + '_state' + '.pt'
    model, input_size = create_model(feature_types, 
                                    inputs = 'all_i',
                                    head = 'poolv1' ,
                                    pretrained = 'imagenet')
    model.load_state_dict(torch.load(path_final, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()

    print("Predict: {}".format(args.partition))
    prediction = {}
    train_per_epoch = len(data_set) // BATCH_SIZE
    if args.partition in ['train','val']:
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

    
    if args.partition in ['train','val']:
        _acc = acc(labels[args.partition], [item for sublist in prediction[args.partition] for item in sublist]) 
        _f1 = f1(labels[args.partition], [item for sublist in prediction[args.partition] for item in sublist])  
        print('{} - acc: {}'.format(args.partition,_acc))
        print('{} - f1: {}'.format(args.partition,_f1))
    else:
        export(prediction[args.partition],test_filenames,data_set,name)

