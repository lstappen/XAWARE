import torch
import os, pickle
import numpy as np
import glob
import utils_images as utils
import random

class DatasetAll(torch.utils.data.Dataset):
    # data.py uses a central configuration
    def __init__(self, data_path, partition, feature_types, testing):

        self.partition = partition
        if partition == 'test':
            self.split_idx = 1
        else:
            self.split_idx = 2

        self.feature_types = feature_types
        self.partition_image_names = self.read_image_names(data_path)

        if testing:
            random.seed(10)
            random.shuffle(self.partition_image_names)
            self.partition_image_names = self.partition_image_names[:50]

        self.load_img = utils.LoadImage()
        self.tf_img = utils.TransformImage() 

        self.features = None
        self.labels = None
        self.label_index = None

        self.load_and_sort_pkl(data_path)

    def load_and_sort_pkl(self, data_path):

        root = '../preprocessed/'

        if self.partition != 'test':
            self.label_index = {self.prediction_classes()[0][i]:i for i in range(self.prediction_classes()[1])} 
            self.labels = self.create_y()
        else:
            self.label_index = {self.prediction_classes()[0][i]:i for i in range(self.prediction_classes()[1])} 

        features = {}
        missing_data = {}
        for feature_type in self.feature_types: 
            
            if feature_type in ['face_extractor','gocar']:
                features_input = []
                path = os.path.join(root,feature_type,'X_' + self.partition + '.pickle')
                temp_data = self.read_pickle(path)
                dim = temp_data[list(temp_data.keys())[0]].shape

                for name in self.partition_image_names:

                    try:
                        features_input.append(np.array(temp_data[name], dtype=float))
                    except KeyError as ke:
                        missing_data.setdefault(feature_type,[]).append(name)
                        features_input.append(np.zeros(dim))

            elif feature_type == 'env_img_extractor':
                features_input = []
                img_names = self.read_images_env(data_path)
                img_cut = ['/'.join(name.split('/')[-self.split_idx:]) for name in img_names]

                for name in self.partition_image_names:
                    index = img_cut.index(name) if name in img_cut else -1
                    if index != -1:
                        features_input.append(img_names[index])
                    else:
                        missing_data.setdefault(feature_type,[]).append(name)
                        features_input.append(None)

            elif feature_type == 'faces_img_extractor':
                features_input = []
                img_names = self.read_images_face(data_path)
                img_cut = ['/'.join(name.split('/')[-self.split_idx:]) for name in img_names]

                for name in self.partition_image_names:
                    index = img_cut.index(name) if name in img_cut else -1
                    if index != -1:
                        features_input.append(img_names[index])
                    else:
                        missing_data.setdefault(feature_type,[]).append(name)
                        features_input.append(None)

            features[feature_type] = features_input

        print('no features existing for: ')
        for k, v in missing_data.items():
            print("{}: {} ".format(k, len(v)))
            print(v)

        length_features = [len(f) for f in features.values()]
        print(length_features)
        if len(list(set(length_features))) > 1:
            print("Not all features have the same number of datapoints")
            print(length_features)
            exit()
     
        self.features = features

    def prediction_classes(self):
        classes = [str(s) for s in [1,2,3,4,5,6,7,8,9]]
        CLASS_NO = len(classes)
        return classes, CLASS_NO


    def create_y(self):

        data_labels=[]
        for file in self.partition_image_names:
            label = file.split('/')[0]  
            data_labels.append(self.label_index[label])
        print('\ndata_labels:\n',set(data_labels))

        data_labels = np.array(data_labels)
        y = data_labels
        print('\nencoded_data_labels:\n',y[:3])
        return y

    def get_real_labels_fromListoLabels(self,label_list):

        index_label = dict((v,k) for k,v in self.label_index.items())
        l = []
        for label_ml in label_list:
            l.append(index_label[label_ml])

        print("transformed:",set(l))

        return l

    def image_path(self,data_path, class_):

        if self.partition == 'test':
            # give back test only once
            if class_ == '1':
                return os.path.join(data_path,self.partition)
            else: 
                return './'
        else:
            return os.path.join(data_path,self.partition,class_)

    def read_images_env(self, data_path):

        data_path = os.path.join(data_path,'env')
        data_files = []
        for class_ in self.prediction_classes()[0]:
            data_files += glob.glob(os.path.join(self.image_path(data_path, class_),'*.png'),recursive=True)+glob.glob(os.path.join(self.image_path(data_path, class_),'*.PNG'),recursive=True)

        num_data_files = len(data_files)
        print('num partition_name:\n',self.partition, num_data_files)

        return data_files

    def read_images_face(self, data_path):

        data_path = os.path.join(data_path,'cropped')
        data_files = []
        for class_ in self.prediction_classes()[0]:
            data_files += glob.glob(os.path.join(self.image_path(data_path, class_),'*.png'),recursive=True)+glob.glob(os.path.join(self.image_path(data_path, class_),'*.PNG'),recursive=True)

        num_data_files = len(data_files)
        print('num partition_name:\n',self.partition, num_data_files)

        return data_files          

    def read_image_names(self, data_path):

        data_path = os.path.join(data_path,'env')      
        data_files = []
        for class_ in self.prediction_classes()[0]:
            data_files += glob.glob(os.path.join(self.image_path(data_path, class_),'*.png'),recursive=True)+glob.glob(os.path.join(self.image_path(data_path, class_),'*.PNG'),recursive=True)

        num_data_files = len(data_files)
        print('num partition_name:\n',self.partition, num_data_files)

        data_files = sorted(['/'.join(name.split('/')[-self.split_idx:]) for name in data_files])

        return data_files

    def one_hot(self, label_array,num_classes):
        return np.squeeze(np.eye(num_classes)[label_array.reshape(-1)])

    def read_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_labels(self):
        return self.labels

    def get_filenames(self):
        return self.partition_image_names

    def __len__(self):
        return len(self.partition_image_names)

    def __getitem__(self, index):
        X = []
        for feature_type in self.feature_types: 
            if 'img' in feature_type:
                if self.features[feature_type][index] is not None:
                    input_img = self.load_img(self.features[feature_type][index])
                    input_tensor = self.tf_img(input_img)
                else:
                    input_tensor = torch.Tensor(3, 299, 299).new_full((3, 299, 299),torch.finfo(torch.float32).eps) #torch.Tensor(3, 299, 299)
                X.append(input_tensor)
            else:
                X.append(self.features[feature_type][index])

        if self.partition == 'test':
            return X
        else:
            return X, self.labels[index]




