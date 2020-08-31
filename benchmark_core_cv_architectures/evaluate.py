import os
import pickle
import csv
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.utils.multiclass import unique_labels


# sklearn cf to visualise the class predictions
def plot_confusion_matrix(y_true, y_pred,
                          normalize=True,
                          title=None,
                          ticklabels=None
                          , path=None
                          , PRECISION_NO=None):
    cmap = cm.Blues
    # Compute confusion matrix
    cma = confusion_matrix(y_true, y_pred)
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cma = cma.astype('float') / cma.sum(axis=1)[:, np.newaxis]
        cma = np.around(cma * 100, decimals=PRECISION_NO)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cma, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax = ax)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    if not ticklabels:
        ticklabels = classes
    ax.set(xticks=np.arange(cma.shape[1]),
           yticks=np.arange(cma.shape[0]),
           xticklabels=ticklabels, yticklabels=ticklabels,
           title=title, ylabel='Truth', xlabel='Prediction')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.1f' if normalize else 'd'
    thresh = cma.max() / 2.
    for i in range(cma.shape[0]):
        for j in range(cma.shape[1]):
            ax.text(j, i, format(cma[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cma[i, j] > thresh else "black")
    fig.tight_layout()

    if path:
        fig.savefig(os.path.join(path, title + '.pdf'))
    return ax


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


# custom metrics
def f1(y_true, y_pred, PRECISION_NO):
    return round(f1_score(y_true, y_pred, average='macro') * 100, PRECISION_NO)


def uar(y_true, y_pred, PRECISION_NO):
    return round(recall_score(y_true, y_pred, average='macro') * 100, PRECISION_NO)


def acc(y_true, y_pred, PRECISION_NO):
    return round(accuracy_score(y_true, y_pred) * 100, PRECISION_NO)


def flatten_score_class(y_true, y_pred, args):
    results = {'f1': f1(y_true, y_pred,  args.PRECISION_NO),
               'uar': uar(y_true, y_pred,  args.PRECISION_NO),
               'acc': acc(y_true, y_pred,  args.PRECISION_NO)}
    return results


def classification_results(log_dir, model, X, y, y_names, args, partition_name):
    print(partition_name)
    model_outputs = model.predict(X, batch_size=args.BATCH_SIZE)
    y_pred = np.argmax(model_outputs, axis=1).tolist()
    y = np.argmax(y, axis=1).tolist()

    report = classification_report(y, y_pred, labels=[i for i in range(args.CLASSES_NO)], target_names=y_names)

    print('classification Report for {} set\n {}'.format(partition_name, report))
    report_df = pd.DataFrame(classification_report(y, y_pred
                                                   , labels=[i for i in range(args.CLASSES_NO)]
                                                   , target_names=y_names
                                                   , output_dict=True
                                                   , digits=2)).transpose().round({'support': 0})
    report_df['support'] = report_df['support'].apply(int)
    report_df.to_csv(os.path.join(log_dir, '{}_classification_report.csv'.format(partition_name)))
    print('classification report for {} exported.'.format(partition_name))

    plot_confusion_matrix(y, y_pred, normalize=True, ticklabels=y_names,
                          title='Confusion matrix {}'.format(partition_name), path=log_dir)
    print('confusion matrix for {} exported.'.format(partition_name))

    results = flatten_score_class(y, y_pred, args)

    with open(os.path.join(log_dir, partition_name + '.csv'), 'w+', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])

    print("  ", partition_name)
    for k, v in results.items():
        print("  - {}: {}".format(k, v))

    export_ys(log_dir, y, y_pred, partition_name)


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
    print(history.history.keys())
    tr_acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(tr_acc) + 1)

    # Train and validation accuracy
    plt.figure()
    plt.plot(epochs, tr_acc, 'b', label='Training accurarcy')
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
    plt.plot(epochs, smooth_plot(tr_acc), 'b', label='Training accurarcy')
    plt.plot(epochs, smooth_plot(val_acc), 'r', label='Validation accurarcy')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'acc_smooth.pdf'))

    plt.figure()
    plt.plot(epochs, smooth_plot(loss), 'b', label='Training loss')
    plt.plot(epochs, smooth_plot(val_loss), 'r', label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss_smooth.pdf'))


def evaluate_model(model, X_train, X_val, y_train, y_val, labels_index, log_dir, args, history):
    sort = sorted(labels_index.items(), key=lambda item: item[1])
    ticklabels = [sort[i][0] for i in range(len(sort))]

    classification_results(log_dir, model, X_train, y_train, ticklabels, args, partition_name='train')
    classification_results(log_dir, model, X_val, y_val, ticklabels, args, partition_name='devel')

    output_graphs(log_dir, history)
