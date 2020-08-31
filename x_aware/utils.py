import random
import numpy as np
import torch
import os
from pathlib import Path
import shutil

from sklearn.utils import class_weight


class EarlyStopping:
    def __init__(self, device, config, test_set, test_generator, monitor='val_loss', mode='auto', verbose=False,
                 delta=0):
        """
        Args:
        """
        print('Set EarlyStopping')
        self.config = config
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = config['best_model_name'] + '.pt'
        self.path_final = config['best_model_name'] + '_state' + '.pt'
        random.seed(1)
        self.monitor = monitor

        self.device = device
        self.test_set = test_set
        self.test_generator = test_generator
        self.test_filenames = test_set.get_filenames()
        self.store_threshold = config['STORE_THRESHOLD']

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'loss' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def __call__(self, score, model, val_acc):

        current = score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        if self.monitor_op(current - self.delta, self.best_score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.config["PATIENCE"]}')
            if self.counter >= self.config['PATIENCE']:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model)
            if val_acc > self.config['STORE_THRESHOLD']:
                print("{} larger than {}".format(val_acc, self.config['STORE_THRESHOLD']))
                y_test_pred = []
                for i, (inputs) in enumerate(self.test_generator):
                    inputs = [i.float().to(self.device) for i in inputs]
                    model.eval()
                    if '_daux' in self.config['head']:
                        outputs, _, _ = model(*inputs)
                    elif '_aux' in self.config['head']:
                        outputs, _ = model(*inputs)
                    else:
                        outputs = model(*inputs)
                    _, y_pred = torch.max(outputs, 1)
                    y_test_pred.append(y_pred.cpu().tolist())

                self.export(y_test_pred, self.test_filenames)

            self.best_score = score
            self.counter = 0

        return self.best_score

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation {self.monitor} ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

    def restore(self, model):
        # load best model weights  
        print("Restore best performing model {}".format(self.best_score))
        # restored = 
        model.load_state_dict(torch.load(self.path))
        torch.save(model, self.path_final)
        return model# restored

    def export(self, test_prediction, test_filenames):
        export_path = os.path.join('../fumo/', self.path.split('/')[-1])

        dirpath = Path(export_path)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)

        if not os.path.exists(export_path):
            os.makedirs(export_path)

        test_flat = [item for sublist in test_prediction for item in sublist]
        test_flat = self.test_set.get_real_labels_fromListoLabels(test_flat)
        for i, file in enumerate(test_filenames):
            with open(os.path.join(export_path, file.split('.')[0] + '.txt'), "w") as file:
                file.write(str(test_flat[i]))


# functions 
def class_weights(y_train):
    return class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)


def export_predictions(test_prediction, test_filenames, data_set, name):
    export_path = os.path.join('../fumo/', name)

    dirpath = Path(export_path)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    test_flat = [item for sublist in test_prediction for item in sublist]
    test_flat = data_set.get_real_labels_fromListoLabels(test_flat)
    print(test_flat)
    for i, file in enumerate(test_filenames):
        with open(os.path.join(export_path, file.split('.')[0] + '.txt'), "w") as file:
            file.write(str(test_flat[i]))


def freeze_unfreeze(UNFREEZE, model, layers_from_bottom, layers_from_top):
    if UNFREEZE:
        print('**Unfreeze mdoel')
    else:
        print('**Freeze model')
    print("if added from front or back")

    if layers_from_bottom > 0:
        print(len(list(model.children())), 'children are available')
        for layer_no, child in enumerate(model.children(), 1):
            if layer_no <= layers_from_bottom:
                print('UNFREEZE: ', UNFREEZE, 'child modules no:', layer_no, len(list(child.parameters())))
                for param in child.parameters():
                    param.requires_grad = UNFREEZE  # if unfreeze = True -> freeze -> set grad to False
            elif layer_no <= layers_from_bottom + 1:
                print('first non affected layer:', child)

    if layers_from_top > 0:
        print(len(list(model.children())), 'children are available')
        for layer_no, child in enumerate(model.children(), 1):
            if layer_no <= layers_from_top:
                print('UNFREEZE: ', UNFREEZE, 'child modules no:', layer_no, len(list(child.parameters())))
                for param in child.parameters():
                    param.requires_grad = UNFREEZE  # if unfreeze = True -> freeze -> set grad to False
            elif layer_no <= layers_from_top + 1:
                print('first non affected layer:', child)

    return model
