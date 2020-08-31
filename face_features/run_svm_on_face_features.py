import pickle, os
from sklearn.metrics import mean_absolute_error, f1_score, recall_score, accuracy_score
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix
import numpy as np

feature_dir = './features'
NORMALISED = False # set if normalised or not

def f1(y_true, y_pred):
    return round(f1_score(y_true, y_pred, average = 'macro')*100, 2) 

def uar(y_true, y_pred):
    return round(recall_score(y_true, y_pred, average = 'macro')*100, 2) 

def acc(y_true, y_pred):
    return round(accuracy_score(y_true, y_pred)*100, 2) 

def flatten_list(l):
    return list(np.array(l).flat)

def measure(y_true, y_pred, name):

    results = {}
    results['f1'] = f1(y_true, y_pred)
    results['uar'] = uar(y_true, y_pred)
    results['acc'] = acc(y_true, y_pred)
    print(results)


def read_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def extract_vector(X):
    features = [y for y in X.values()]
    feature_vectors = []
    for line in features:
        feature_vector = [v for k,v in line.items()]
        feature_vector =  [item for sublist in feature_vector for item in sublist]
        feature_vectors.append([i if i is not None else 0 for i in feature_vector])
    return feature_vectors

def run_svm(X_train, X_devel, X_test, y_train, y_devel, name):
    y_train = y_train.tolist()
    y_devel = y_devel.tolist()
    
    print(name)
    complexities = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1] #

    # Train SVM model with different complexities and evaluate
    acc_scores = []
    for comp in complexities:
        print('\nComplexity {0:.6f}'.format(comp))
        clf = svm.LinearSVC(C=comp, random_state=0, max_iter=100000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_devel)
        measure(y_devel, y_pred, 'devel')
        acc_scores.append(accuracy_score(y_devel , y_pred))
        print('ACC on Devel {0:.1f}'.format(acc_scores[-1] * 100))
        
if __name__== "__main__":
    if NORMALISED:
        X_train = read_data(os.path.join(feature_dir,'X_'+'train'+'.pickle'))
        y_train = read_data(os.path.join(feature_dir,'y_'+'train'+'.pickle'))

        X_devel = read_data(os.path.join(feature_dir,'X_'+'val'+'.pickle'))
        y_devel = read_data(os.path.join(feature_dir,'y_'+'val'+'.pickle'))

        X_test = read_data(os.path.join(feature_dir,'X_'+'test'+'.pickle'))
    else:
        X_train = np.array(extract_vector(read_data(os.path.join(feature_dir,'X_raw_features_'+'train'+'.pickle'))))
        y_train = read_data(os.path.join(feature_dir,'y_'+'train'+'.pickle'))

        X_devel = np.array(extract_vector(read_data(os.path.join(feature_dir,'X_raw_features_'+'val'+'.pickle'))))
        y_devel = read_data(os.path.join(feature_dir,'y_'+'val'+'.pickle'))

        X_test = np.array(extract_vector(read_data(os.path.join(feature_dir,'X_raw_features_'+'test'+'.pickle'))))  


    run_svm(X_train, X_devel, X_test, y_train, y_devel, 'normalised')

    run_svm(X_train, X_devel, X_test, y_train, y_devel, 'not_normalised')