import cv2
from gaze_tracking import GazeTracking
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import argparse, glob, pickle, os
import numpy as np

parser = argparse.ArgumentParser(description = 'Face feature extractor')
# raw data paths
parser.add_argument('-d', '--data_path', type = str, dest = 'data_path', required = False, action = 'store', 
                    default = '../data/env/', 
                    help = 'specify which data path')
parser.add_argument('-type', '--extraction_type', type = str, dest = 'extraction_type', required = False, action = 'store', 
                    default = 'mixed', # or cnn 
                    help = 'specify if mixed (hog + cnn) or purely cnn extraction')
args = parser.parse_args()

# recalibrate over entire data set
gaze = GazeTracking(args.extraction_type)


def prediction_classes():
    classes = [str(s) for s in [0,1,2,3,4,5,6,7,8,9]]
    CLASS_NO = len(classes)
    print('%s of classes are going to be predicted'%CLASS_NO)
    return classes, CLASS_NO

def one_hot(label_array,num_classes):
   return np.squeeze(np.eye(num_classes)[label_array.reshape(-1)])

def read_and_extract_facefeatures(dir_of_images, nrows = 640, ncolumns = 480):

    if args.extraction_type =='mixed':
        add = ''
    else:
        add = args.extraction_type
    
    # examples
    # open eyes no glasses: Sub27_vid_1_frame23.png
    # closed eyes glasses: Sub27_vid_2_frame31.png
    
    X_f = [] # list of features
    X = {}
    for i, path in enumerate(dir_of_images):
        print(i,path)
        cropped_face_filename = os.path.join("/".join(path.split('/')[:2]),'cropped'+add,"/".join(path.split('/')[2:]))
        if not os.path.exists("/".join(cropped_face_filename.split('/')[:-1])):
            os.makedirs("/".join(cropped_face_filename.split('/')[:-1]))
        frame = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC)
        gaze.refresh(frame,cropped_face_filename)
        features, feature_vector = gaze.features()        

        X_f.append(feature_vector)
        X[(i, path)] = features

    gaze.close_run()
    return X_f, X

def prepare_data(feature_dir, CLASSES, CLASSES_NO):

    label_index = {CLASSES[i]:i for i in range(CLASSES_NO)} 
    labels_index = {CLASSES[i]+'_{}'.format(i):i for i in range(CLASSES_NO)} # with suffix for convenience
    print('label_index:\n',labels_index)


    def read_data(partition_name):

        data_files = []
        if partition_name != 'test':
            for class_ in CLASSES:
                data_files += glob.glob(os.path.join(args.data_path,partition_name,class_,'*.png'),recursive=True)+glob.glob(os.path.join(args.data_path,partition_name,class_,'*.PNG'),recursive=True)
        else:
            data_files += glob.glob(os.path.join(args.data_path,partition_name,'*.png'),recursive=True)+glob.glob(os.path.join(args.data_path,partition_name,'*.PNG'),recursive=True)
        # Introduce random into calibration
        #shuffle(data_files)
        #data_files = data_files[21880:]
        

        num_data_files = len(data_files)
        print('num partition_name:\n',partition_name,data_files[:3])

        if partition_name != 'test':
            data_labels=[]
            for file in data_files:
                label = file.split('/')[-2]  # adjust according to form of dir
                data_labels.append(label_index[label])
            print('\ndata_labels:\n',data_labels[:3])
            assert num_data_files == len(data_labels)
            data_labels = np.array(data_labels)
            y_hot = one_hot(data_labels,CLASSES_NO)
            print('\nencoded_data_labels:\n',y_hot[:3])

        X_f, X = read_and_extract_facefeatures(data_files)

        X_f = np.array(X_f)
        normalizedX = X_f / X_f.max(axis=0)
        print('\ndata_features:\n',normalizedX[:1])    

        print("export")    
        with open(os.path.join(feature_dir,'X_raw_features_'+partition_name+'.pickle'), 'wb') as handle:
            pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(feature_dir,'X_'+partition_name+'.pickle'), 'wb') as handle:
            pickle.dump(normalizedX, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if partition_name != 'test':
            with open(os.path.join(feature_dir,'y_'+partition_name+'.pickle'), 'wb') as handle:
                pickle.dump(np.array(data_labels), handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(feature_dir,'y_hot'+partition_name+'.pickle'), 'wb') as handle:
                pickle.dump(np.array(y_hot), handle, protocol=pickle.HIGHEST_PROTOCOL)

        if partition_name != 'test':
            return normalizedX.shape, y_hot.shape
        else:
            return normalizedX.shape, None       

    X_train, y_train = read_data('train')
    X_val,y_val = read_data('val')
    X_test,_ = read_data('test')

    print("Shape of train images is:", X_train)
    print("Shape of validation images is:", X_val)
    print("Shape of test images is:", X_test)
    print("Shape of labels is:", y_train)
    print("Shape of labels is:", y_val)

    return X_train, X_val, y_train, y_val, labels_index

if __name__  == "__main__":

    feature_dir = './features'
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    CLASSES, CLASSES_NO = prediction_classes()

    X_train, X_val, y_train, y_val, labels_index = prepare_data(feature_dir, CLASSES, CLASSES_NO)