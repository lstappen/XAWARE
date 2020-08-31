import random
random.seed(30)
import matplotlib
import numpy as np
import pandas as pd
import os
import csv
import torch
import pickle
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix #, precision_score
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from data import prediction_classes
matplotlib.use('Agg')

PRECISION_NO = 4
y_names, CLASSES_NO = prediction_classes()

# custom metrics
def f1(y_true, y_pred):
    return round(f1_score(y_true, y_pred, average='macro') * 100, PRECISION_NO)


def uar(y_true, y_pred):
    return round(recall_score(y_true, y_pred, average='macro') * 100, PRECISION_NO)


def acc(y_true, y_pred):
    return round(accuracy_score(y_true, y_pred) * 100, PRECISION_NO)


def smooth_plot(points, factor=0.7):
    smooth_pts = []
    for point in points:
        if smooth_pts:
            previous = smooth_pts[-1]
            smooth_pts.append(previous * factor + point * (1 - factor))
        else:
            smooth_pts.append(point)

    return smooth_pts


def output_graphs(log_dir, history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Train and validation accuracy
    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'acc.pdf'))

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss.pdf'))

    plt.figure()
    plt.plot(epochs, smooth_plot(acc), 'b', label='Training accurarcy')
    plt.plot(epochs, smooth_plot(val_acc), 'r', label='Validation accurarcy')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'acc_smooth.pdf'))

    plt.figure()
    plt.plot(epochs, smooth_plot(loss), 'b', label='Training loss')
    plt.plot(epochs, smooth_plot(val_loss), 'r', label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss_smooth.pdf'))


# sklearn cf to visualise the class predictions
def plot_confusion_matrix(y_true, y_pred,
                          normalize=True,
                          title=None,
                          ticklabels=None
                          , path=None
                          , cmap=plt.cm.Blues):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm * 100, decimals=PRECISION_NO)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    if not ticklabels:
        ticklabels = classes
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=ticklabels, yticklabels=ticklabels,
           title=title, ylabel='Truth', xlabel='Prediction')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if path:
        fig.savefig(os.path.join(path, title + '.pdf'))
    return ax


def flatten_score_class(y_true, y_pred):
    results = {'f1': f1(y_true, y_pred), 'uar': uar(y_true, y_pred), 'acc': acc(y_true, y_pred)}
    return results


def export_ys(log_dir, y, y_pred, partition_name):
    y_path = os.path.join(log_dir, 'y')
    y_pred_path = os.path.join(log_dir, 'y_pred')

    if not os.path.exists(y_path):
        os.makedirs(y_path)
    if not os.path.exists(y_pred_path):
        os.makedirs(y_pred_path)

    with open(os.path.join(y_pred_path, "{}.pkl".format(partition_name)), "wb") as f:
        pickle.dump(y_pred, f)
    with open(os.path.join(y_path, "{}.pkl".format(partition_name)), "wb") as f:
        pickle.dump(y, f)


def classification_results(config, y, y_pred, partition_name):

    report = classification_report(y, y_pred, labels=[i for i in range(CLASSES_NO)], target_names=y_names)

    print('classification Report for {} set\n {}'.format(partition_name, report))
    report_df = pd.DataFrame(classification_report(y, y_pred
                                                   , labels=[i for i in range(CLASSES_NO)]
                                                   , target_names=y_names
                                                   , output_dict=True
                                                   , digits=2)).transpose().round({'support': 0})
    report_df['support'] = report_df['support'].apply(int)
    report_df.to_csv(os.path.join(config['log_dir'], '{}_classification_report.csv'.format(partition_name)))
    print('classification report for {} exported.'.format(partition_name))

    plot_confusion_matrix(y, y_pred, normalize=True, ticklabels=y_names,
                          title='Confusion matrix {}'.format(partition_name), path=config['log_dir'])
    print('confusion matrix for {} exported.'.format(partition_name))

    results = flatten_score_class(y, y_pred)

    with open(os.path.join(config['log_dir'], partition_name + '.csv'), 'w+', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])

    print("  ", partition_name)
    for k, v in results.items():
        print("  - {}: {}".format(k, v))

    export_ys(config['log_dir'], y, y_pred, partition_name)


def evaluate_model(model, dataloader, dataset_sizes, device, partition_name, config):

    model.eval()
    prediction, y = [], []
    for i, (inputs, ys) in enumerate(dataloader):
        print("Progress {:2.1%} ".format(i / dataset_sizes[partition_name]), end="\r")
        inputs = [i.float().to(device) for i in inputs]
        outputs = model(*inputs)
        _, y_pred = torch.max(outputs, 1)
        prediction.append(y_pred.cpu().tolist())
        y.append(ys.cpu().tolist())

    prediction = [item for sublist in prediction for item in sublist]
    y = [item for sublist in y for item in sublist]

    classification_results(config, y, prediction, partition_name)
