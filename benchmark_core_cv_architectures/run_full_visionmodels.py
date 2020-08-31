import argparse
import matplotlib
import os
import random

import matplotlib.pyplot as plt
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from StepsCallbacks import StepsEarlyStopping, Metrics
from config import set_config
from data import prepare_data
from evaluate import evaluate_model
from model import input_size, create_model, class_weights

random.seed(30)

# os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # specify which GPU(s) to be used

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='Vision model evaluation')
# raw data paths
parser.add_argument('-d', '--data_path', type=str, dest='data_path', required=False, action='store',
                    default='../data/',
                    help='specify which data path')
parser.add_argument('-e', '--experiments_path', type=str, dest='experiments_path', required=False, action='store',
                    default="./experiments/", help='specify the data folder')
parser.add_argument('-m', '--model_name', type=str, dest='model_name', required=False, action='store',
                    default='all', help='specify model name.')
# 'ResNet50', 'VGG-16', 'VGG-19', 'Xception', 'InceptionResNetV2', 'InceptionV3'
parser.add_argument('-c', '--cropped_faces', dest='cropped_faces', required=False, action='store_true'
                    , help='specify if face images (true) or the whole pictures should be used')
parser.add_argument('-ld', '--lr_decay', dest='lr_decay', required=False, action='store_true'
                    , help='specify if learning rate decay')
parser.add_argument('-t', '--testing_switch', dest='testing_switch', required=False,
                    action='store_true', help='specify if model should be run with few data')
parser.add_argument('-he', '--head', type=str, dest='head', required=False, action='store',
                    default="t_complex", help='specify if the top layers of the model')
parser.add_argument('-base', '--base_trainable', dest='base_trainable', required=False, type=bool,
                    default=True,
                    help='specify if the base model should be trainable, thus, potential preloaded weights are fine-tuned.')
args_input = parser.parse_args()

if __name__ == "__main__":

    args = set_config(args_input)

    if args.TEST:
        model_names = ['ResNet50']
    elif args.model_name == 'all':
        model_names = ['VGG16', 'ResNet50', 'VGG19', 'InceptionV3', 'Xception', 'InceptionResNetV2']
    else:
        model_names = [args.model_name]

    input_shape_old = (1, 1, 1)

    for model_name in model_names:

        input_shape = input_size(model_name, args)

        name = ''
        if args.cropped_faces:
            name = name + 'faces_'
        else:
            name = name + 'env_'
        name = name + model_name

        if args.base_trainable:
            name = name + '_btrainable'
        else:
            name = name + '_bfrozen'
        name = name + '_' + args.head
        if args.lr_decay:
            name = name + '_lrdecay'
        name = name + '_' + str(args.CLASSES_NO)
        if args.testing_switch:
            name = name + '_T'

        log_dir = os.path.join(args.experiments_path, name)
        best_model_dir = os.path.join(args.experiments_path, 'best_models')
        best_model_name = os.path.join(best_model_dir, name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        if input_shape != input_shape_old:
            X_train, X_val, y_train, y_val, labels_index = prepare_data(input_shape, args)
        else:
            X_train, X_val, y_train, y_val, labels_index = None, None, None, None, None
            exit(1)

        model = create_model(model_name, log_dir, args)

        metric = Metrics((X_val, y_val), args.PRECISION_NO)
        callbacks_list = []
        if args.lr_decay:
            steps_callback = StepsEarlyStopping(filepath=best_model_name  # without .h5
                                                , check_interval=args.DECAY_STEPS
                                                , val_data=(X_val, y_val)
                                                , monitor='f1'  # 'val_loss'
                                                , patience=args.PATIENCE)
            callbacks_list = [metric, steps_callback]
        else:
            early_stopping = EarlyStopping(monitor='val_acc', patience=args.PATIENCE, verbose=1)
            checkpoint = ModelCheckpoint(best_model_name,
                                         monitor='val_acc', save_weights_only=False, save_best_only=True,
                                         mode='auto')  # , period=3
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=3, min_lr=0.000001)
            callbacks_list = [metric, early_stopping, checkpoint, reduce_lr]

        history = model.fit(X_train, y_train, batch_size=args.BATCH_SIZE
                            , epochs=args.EPOCHS
                            , validation_data=(X_val, y_val)
                            , callbacks=callbacks_list  # callbacks_list
                            , class_weight=class_weights(y_train))

        # model.save(os.path.join(log_dir,'car_parts.h5'))
        del model
        model = load_model(best_model_name)

        evaluate_model(model, X_train, X_val, y_train, y_val, labels_index, log_dir, args, history)
        input_shape_old = input_shape
        # Collect the garbage
        del model
        backend.clear_session()
        plt.close('all')
