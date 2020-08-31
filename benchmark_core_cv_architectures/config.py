from argparse import Namespace
from data import prediction_classes
import argparse


def set_config(args_input):
    print(args_input)
    args = argparse.Namespace()
    dict_args = vars(args_input)

    TEST = False
    if args.testing_switch:
        TEST = True
        print('-' * 30)
        print("RUN IN TEST-MODE!")
        print('-' * 30)

    if args.lr_decay:
        print("Using learning rate decay")
        INIT_LEARN_RATE = 0.0001
        DECAY_RATE = 0.96
        DECAY_STEPS = 100
        LEARNING_RATE = None
    else:
        LEARNING_RATE = 0.0001
        DECAY_STEPS = None
        DECAY_RATE = None
        INIT_LEARN_RATE = None

    PATIENCE = 3
    PRECISION_NO = 2
    channels = 3  # change to 1 if you want to use grayscale image
    EPOCHS = 400  # 50

    BATCH_SIZE = 64

    if TEST:
        BATCH_SIZE = 16
        DECAY_STEPS = 3
    if args.model_name == 'NASNetLarge':
        print("reduce bs to avoid memory overflow")
        BATCH_SIZE = 16  # memory overflow

    CLASSES, CLASSES_NO = prediction_classes()

    external_args = {
        'TEST': TEST
        , 'INIT_LEARN_RATE': INIT_LEARN_RATE
        , 'LEARNING_RATE': LEARNING_RATE
        , 'DECAY_RATE': DECAY_RATE
        , 'DECAY_STEPS': DECAY_STEPS
        , 'PATIENCE': PATIENCE
        , 'PRECISION_NO': PRECISION_NO
        , 'channels': channels
        , 'EPOCHS': EPOCHS
        , 'BATCH_SIZE': BATCH_SIZE
        , 'CLASSES': CLASSES
        , 'CLASSES_NO': CLASSES_NO
    }

    return Namespace(**{**dict_args, **external_args})
