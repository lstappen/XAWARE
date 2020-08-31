import pickle, os
import numpy as np

feature_dir = './features'
export_dir = '../preprocessed/face_extractor'

def read_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def extract_vector(X):
    features=[]
    meta=[]
    for k,v in X.items():
        features.append(v)
        meta.append(k)
    feature_vectors = []
    for line in features:
        feature_vector = [v for k,v in line.items()]
        feature_vector =  [item for sublist in feature_vector for item in sublist]
        feature_vectors.append([i if i is not None else 0 for i in feature_vector])
    return feature_vectors, meta


def export(X_N, meta, partition_name):
    export_format = {}
    if partition_name == 'test':
        idx = 1
    else:
        idx = 2
    for i in range(len(X_N)):
        
        filename_key = "/".join(meta[i][1].split('/')[-idx:])
        export_format[filename_key] = X_N[i]

    print(list(export_format.keys())[0])
    print(export_format[list(export_format.keys())[0]])
    
    with open(os.path.join(export_dir,'X_'+partition_name+'.pickle'), 'wb') as handle:
        pickle.dump(export_format, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__== "__main__":
    NORMALISED = False


    X_train_N = read_data(os.path.join(feature_dir,'X_'+'train'+'.pickle'))
    y_train_N = read_data(os.path.join(feature_dir,'y_'+'train'+'.pickle'))

    X_devel_N = read_data(os.path.join(feature_dir,'X_'+'val'+'.pickle'))
    y_devel_N = read_data(os.path.join(feature_dir,'y_'+'val'+'.pickle'))

    X_test_N = read_data(os.path.join(feature_dir,'X_'+'test'+'.pickle'))

    train, train_meta = extract_vector(read_data(os.path.join(feature_dir,'X_raw_features_'+'train'+'.pickle')))
    X_train = np.array(train)
    y_train = read_data(os.path.join(feature_dir,'y_'+'train'+'.pickle'))

    devel, devel_meta = extract_vector(read_data(os.path.join(feature_dir,'X_raw_features_'+'val'+'.pickle')))
    X_devel = np.array(devel)
    y_devel = read_data(os.path.join(feature_dir,'y_'+'val'+'.pickle'))

    test, test_meta = extract_vector(read_data(os.path.join(feature_dir,'X_raw_features_'+'test'+'.pickle')))
    X_test = np.array(test)  


    export(X_train_N, train_meta, partition_name='train')
    export(X_devel_N, devel_meta, partition_name='val')
    export(X_test_N, test_meta, partition_name='test')
