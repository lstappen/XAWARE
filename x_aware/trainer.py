import torch
import time
from metrics import f1, acc
from utils import freeze_unfreeze


def train_model(device, config, model, dataloaders, dataset_sizes, criterion, optimizer
                , scheduler, name, early_stopping, train_per_epoch):
    since = time.time()
    best_score = 0.0
    history = {}

    for epoch in range(config['MAX_EPOCHS']):

        print('Epoch {}/{}'.format(epoch, config['MAX_EPOCHS'] - 1))

        if config['FREEZE'] is not None:
            if config['FREEZE']['epochs'] == epoch:
                model = freeze_unfreeze(False, model, config['FREEZE']['layers_from_bottom'],
                                        config['FREEZE']['layers_from_top'])

        # For each epoch run training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_labels = []
            running_preds = []

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = [i.float().to(device) for i in inputs]
                labels = labels.long().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    if '_aux' in config['head']:
                        outputs, aux = model(*inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss2 = criterion(aux, labels)

                        loss_total = 0.6 * loss + 0.4 * loss2
                    elif '_daux' in config['head']:
                        outputs, aux1, aux2 = model(*inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss2 = criterion(aux1, labels)
                        loss3 = criterion(aux2, labels)

                        loss_total = loss + 0.8 * loss2 + 0.1 * loss2
                    else:
                        outputs = model(*inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss_total = loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()

                        if config['FREEZE'] is not None:
                            if epoch + 1 == config['FREEZE']['epochs']:
                                model = freeze_unfreeze(False, model, config['FREEZE']['layers_from_bottom'],
                                                        config['FREEZE']['layers_from_top'])

                        if config['UNFREEZE'] is not None:
                            if epoch + 1 == config['UNFREEZE']['epochs']:
                                model = freeze_unfreeze(True, model, config['UNFREEZE']['layers_from_bottom'],
                                                        config['UNFREEZE']['layers_from_top'])

                        if config['batch_lr_adjustment'] is not None:
                            try:
                                steps = config['batch_lr_adjustment']['defined'][epoch + 1]
                            except KeyError:
                                steps = config['batch_lr_adjustment']['steps_non_defined']
                            if i + 1 % steps == 0:
                                print(epoch, "Reduce scheduler at step {}".format(i))
                                scheduler.step()
                            else:
                                print(i + 1 % steps == 0)

                # store measures for averaging
                running_loss += loss_total.item() * inputs[0].size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_labels.append(labels.data.cpu().tolist())
                running_preds.append(preds.cpu().tolist())

                # output during epoch
                if phase == 'train':
                    if '_aux' in name:
                        print("Progress {:2.1%} - L1: {:8.4} L2: {:8.4}".format(i / train_per_epoch, float(loss.item()),
                                                                                float(loss2.item())), end="\r")
                    elif '_doubleaux' in name:
                        print("Progress {:2.1%} - L All: {:8.4} L2: {:8.4}  L3: {:8.4}".format(i / train_per_epoch,
                                                                                               float(loss.item()),
                                                                                               float(loss2.item()),
                                                                                               float(loss3.item())),
                              end="\r")
                    else:
                        print("Progress {:2.1%} - {:8.4}".format(i / train_per_epoch, float(loss.item())), end="\r")  #

            if config['batch_lr_adjustment'] is None:
                print("Reduce scheduler at epoch {}".format(epoch))
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = acc([item for sublist in running_labels for item in sublist],
                            [item for sublist in running_preds for item in sublist])
            epoch_f1 = f1([item for sublist in running_labels for item in sublist],
                          [item for sublist in running_preds for item in sublist])
            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1))

            # store previous run
            if phase == 'train':
                history.setdefault('acc', []).append(epoch_acc)
                history.setdefault('f1', []).append(epoch_f1)
                history.setdefault('loss', []).append(round(epoch_loss, 4))

            if phase == 'val':  # and epoch_acc > best_acc
                history.setdefault(phase + '_acc', []).append(epoch_acc)
                history.setdefault(phase + '_f1', []).append(epoch_f1)
                history.setdefault(phase + '_loss', []).append(round(epoch_loss, 4))

                # early_stopping
                # needs a measure to check if it has decresed, 
                # and if so, make a checkpoint of the current model
                if config['TRACKING_MEASURE'] == 'val_acc':
                    best_score = early_stopping(epoch_acc, model, epoch_acc)
                elif config['TRACKING_MEASURE'] == 'val_loss':
                    best_score = early_stopping(epoch_acc, model, epoch_loss)
                else:
                    print("Tracking measure {} not known".format(config['TRACKING_MEASURE']))
                    exit()

        if early_stopping.early_stop:
            print("Early stopping")
            model = early_stopping.restore(model)
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best score: {:4f}'.format(best_score))

    return model, history
