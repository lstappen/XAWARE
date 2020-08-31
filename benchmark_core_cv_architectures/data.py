import glob
import os
import cv2
import numpy as np
from random import shuffle


# data
def one_hot(label_array, num_classes):
    return np.squeeze(np.eye(num_classes)[label_array.reshape(-1)])


def read_and_resize_image(dir_of_images, input_shape):
    nrows, ncolumns, _ = input_shape

    X = []  # list of images
    for path in dir_of_images:
        X.append(cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
    return X


def prepare_data(input_shape, args):
    label_index = {args.CLASSES[i]: i for i in range(args.CLASSES_NO)}
    labels_index = {args.CLASSES[i] + '_{}'.format(i): i for i in range(args.CLASSES_NO)}  # with suffix for convenience
    print('label_index:\n', labels_index)

    def read_data(partition_name, args):

        if args.cropped_faces:
            # faces
            data_path = os.path.join(args.data_path, 'cropped')
        else:
            data_path = os.path.join(args.data_path, 'env')
            # evn

        data_files = []
        for class_ in args.CLASSES:
            data_files += glob.glob(os.path.join(data_path, partition_name, class_, '*.png'),
                                    recursive=True) + glob.glob(
                os.path.join(data_path, partition_name, class_, '*.PNG'), recursive=True)

        # Fiter out wrong classes
        shuffle(data_files)

        num_data_files = len(data_files)
        print('num partition_name:\n', partition_name, data_files[:3])

        data_labels = []
        for file in data_files:
            label = file.split('/')[-2]  # adjust according to form of dir
            data_labels.append(label_index[label])
        print('\ndata_labels:\n', data_labels[:3])
        assert num_data_files == len(data_labels)

        data_labels = np.array(data_labels)
        y = one_hot(data_labels, args.CLASSES_NO)
        print('\nencoded_data_labels:\n', y[:3])

        if args.TEST:
            X = read_and_resize_image(data_files[:200], input_shape)
        else:
            X = read_and_resize_image(data_files, input_shape)
        X = [img[:, :, [2, 1, 0]] for img in X]  # BGR --> RGB

        X = np.array(X) / 255  # range(0,255) --> (0,1)
        if args.TEST:
            y = np.array(y[:200])
        else:
            y = np.array(y)

        return X, y

    X_train, y_train = read_data('train', args)
    X_val, y_val = read_data('val', args)

    print("Shape of train images is:", X_train.shape)
    print("Shape of validation images is:", X_val.shape)
    print("Shape of labels is:", y_train.shape)
    print("Shape of labels is:", y_val.shape)

    return X_train, X_val, y_train, y_val, labels_index


def prediction_classes():
    classes = [str(s) for s in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    CLASS_NO = len(classes)
    print('%s of classes are going to be predicted' % CLASS_NO)
    return classes, CLASS_NO
