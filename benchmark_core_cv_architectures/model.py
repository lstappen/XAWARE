import numpy as np
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, InceptionV3, Xception, VGG16, VGG19, \
    NASNetMobile, NASNetLarge, DenseNet201, MobileNetV2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.utils import class_weight
from contextlib import redirect_stdout


def input_size(model_name, args):
    if model_name in ('VGG16', 'VGG19', 'ResNet50', 'NASNetMobile', 'DenseNet201', 'MobileNetV2'):
        inputShape = (224, 224, args.channels)
    elif model_name in ('InceptionV3', 'Xception', 'InceptionResNetV2'):
        inputShape = (299, 299, args.channels)
    elif model_name in ['NASNetLarge']:
        inputShape = (331, 331, args.channels)
    else:
        inputShape = (224, 244, args.channels)

    return inputShape


def create_model(model_name, log_dir, args):  # optimizer, learning rate, activation, neurons, batch size, epochs...

    input_shape = input_size(model_name, args)

    if args.head == 'max' or (args.base_trainable and args.head != 't_complex'):
        pool = 'max'
    else:
        pool = 'none'

    if model_name == 'VGG16':
        conv_base = VGG16(weights='imagenet', include_top=False, pooling=pool, input_shape=input_shape)
    elif model_name == 'VGG19':
        conv_base = VGG19(weights='imagenet', include_top=False, pooling=pool, input_shape=input_shape)
    elif model_name == 'ResNet50':
        conv_base = ResNet50(weights='imagenet', include_top=False, pooling=pool, input_shape=input_shape)
    elif model_name == 'InceptionV3':
        conv_base = InceptionV3(weights='imagenet', include_top=False, pooling=pool, input_shape=input_shape)
    elif model_name == 'Xception':
        conv_base = Xception(weights='imagenet', include_top=False, pooling=pool, input_shape=input_shape)
    elif model_name == 'InceptionResNetV2':
        conv_base = InceptionResNetV2(weights='imagenet', include_top=False, pooling=pool, input_shape=input_shape)
    elif model_name == 'NASNetMobile':
        conv_base = NASNetMobile(weights='imagenet', include_top=False, pooling=pool, input_shape=input_shape)
    elif model_name == 'NASNetLarge':
        conv_base = NASNetLarge(weights='imagenet', include_top=False, pooling=pool, input_shape=input_shape)
    elif model_name == 'DenseNet201':
        conv_base = DenseNet201(weights='imagenet', include_top=False, pooling=pool, input_shape=input_shape)
    elif model_name == 'MobileNetV2':
        conv_base = MobileNetV2(weights='imagenet', include_top=False, pooling=pool, input_shape=input_shape)
    else:
        conv_base = None
        print("Model name not known!")
        exit()

    conv_base.trainable = args.base_trainable

    model = models.Sequential()
    if args.base_trainable:
        if args.head == 't_complex':
            model = models.Sequential()
            model.add(conv_base)
            model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', strides=1))
            model.add(layers.Flatten())  # ??
            model.add(layers.Dense(1024, activation='sigmoid'))
            model.add(layers.Dense(256, activation='sigmoid'))
            model.add(layers.Dense(args.CLASSES_NO, activation='softmax'))  # (samples, new_rows, new_cols, filters)
        else:
            model.add(conv_base)
            model.add(layers.Dense(args.CLASSES_NO, activation='softmax'))
    elif args.head == 'dense':
        # outside only?
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(args.CLASSES_NO, activation='softmax'))
    elif args.head == 'max':
        model.add(conv_base)
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(args.CLASSES_NO, activation='softmax'))
    elif args.head == 'mod':
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Conv2D(filters=2048, kernel_size=(3, 3), padding='valid'))
        model.add(layers.Flatten())  # ??
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='sigmoid'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(args.CLASSES_NO, activation='softmax'))  # (samples, new_rows, new_cols, filters)

    if args.lr_decay:
        lr_schedule = ExponentialDecay(
            args.INIT_LEARN_RATE,
            decay_steps=args.DECAY_STEPS,
            decay_rate=args.DECAY_RATE,
            staircase=True)
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr_schedule),
                      metrics=['acc'])  # To different optimisers?
    else:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.LEARNING_RATE), metrics=['acc'])

    with open(os.path.join(log_dir, 'modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    print(model.summary())
    return model


def class_weights(y_train):
    y_train_decoded = np.argmax(y_train, axis=1)
    weights_list = class_weight.compute_class_weight('balanced', np.unique(y_train_decoded), y_train_decoded)
    weights = dict(
        zip(list(np.unique(y_train_decoded)), weights_list))  # careful! list also makes difference, but not the correct
    return weights
