import random

import numpy as np
from sklearn.metrics import f1_score, recall_score
from tensorflow.keras.callbacks import Callback


class StepsEarlyStopping(Callback):
    def __init__(self
                 , filepath  # without .h5
                 , check_interval
                 , val_data
                 , monitor='val_loss'
                 , patience=0
                 , min_delta=0
                 , verbose=1
                 , mode='auto'
                 , no_val_selection=None
                 ):
        super(StepsEarlyStopping, self).__init__()
        self.best = None
        self.model = None
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.filepath = filepath

        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_batch = 0
        self.steps = 0
        self.check_interval = check_interval
        random.seed(1)
        self.monitor = monitor

        if no_val_selection is None:
            # relative 10% of val
            no_val_selection = int(len(val_data[0]) * 0.1)
            print(no_val_selection)

        if self.monitor == 'f1':
            mode = 'max'
        self.validation_data = val_data
        self.val_random_selection = [random.randint(0, len(val_data[0])) for iter in range(no_val_selection)]

        if mode not in ['auto', 'min', 'max']:
            print('BatchEarlyStopping mode %s is unknown, '
                  'fallback to auto mode.' % mode)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.best_weights = None

    def on_train_begin(self, **kwargs):
        self.wait = 0
        self.stopped_batch = 0
        self.steps = 0
        self.best_weights = None

    def set_model(self, model):
        self.model = model

    def on_batch_end(self, batch, logs=None):

        if (self.steps % self.check_interval) == 0:
            print("Calc val metric")
            if self.monitor == 'f1':
                y_pred = (np.asarray(self.model.predict(self.validation_data[0][self.val_random_selection]))).round()
                y_true = self.validation_data[1][self.val_random_selection]
                _val_f1 = round(f1_score(y_true, y_pred, average='macro') * 100, 4)

                print("step %s a F1 %s on val sample: " % (self.steps, _val_f1))
                current = _val_f1  # logs.get(self.monitor)
            else:
                current = logs.get(self.monitor)


            if current is None:
                print(
                    'Batch Early stopping conditioned on metric `%s` '
                    'which is not available. Available metrics are: %s' %
                    (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
                )
                return

            if self.monitor_op(current - self.min_delta, self.best):
                self.wait = 0
                print('Store model at step: ', self.steps)
                self.model.save(self.filepath + '_' + str(self.steps) + '.h5', overwrite=True)
                self.best = current
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_batch = batch
                    self.model.stop_training = True

        self.steps += 1

    def on_train_end(self, **kwargs):
        if self.stopped_batch > 0 and self.verbose > 0:
            print('Steps %05d: early stopping' % (self.steps + 1))


class Metrics(Callback):

    def __init__(self, val_data, PRECISION_NO):
        super().__init__()
        self.val_f1s = []
        self.val_recalls = []
        self.validation_data = val_data
        self.PRECISION_NO = PRECISION_NO

    def f1(self, y_true, y_pred):
        return round(f1_score(y_true, y_pred, average='macro') * 100, self.PRECISION_NO)

    def uar(self, y_true, y_pred):
        return round(recall_score(y_true, y_pred, average='macro') * 100, self.PRECISION_NO)

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_pred = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_true = self.validation_data[1]

        _val_f1 = self.f1(val_true, val_pred)
        _val_recall = self.uar(val_true, val_pred)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)

        print("— val_f1: %f — val_recall %f" % (_val_f1, _val_recall))
        return
