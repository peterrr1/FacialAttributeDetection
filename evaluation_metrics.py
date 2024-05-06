from sklearn import metrics
import numpy as np


def hamming_score(y_true, y_pred):
    """
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857

    Parameters
    ----------
    y_true : 2D array
        Ground truth (correct) target values.
    y_pred : 2D array
        Estimated targets as returned by a classifier.

    Returns
    -------
    float
        Hamming score (float in [0.0, 1.0])

    Notes
    -----
    The Hamming score is defined as the fraction of the true positive predictions 
    over the union of the predicted and true labels.
    """
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])

        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def partial_accuracy_score(y_true, y_pred):
    """
    Compute the partial accuracy score for multi-label classification.

    Parameters
    ----------
    y_true : 2D array
        Ground truth (correct) target values.
    y_pred : 2D array
        Estimated targets as returned by a classifier.

    Returns
    -------
    float
        Partial accuracy score (float in [0.0, 1.0])

    Notes
    -----
    The partial accuracy score is defined as the number of correct predictions 
    divided by the total number of predictions.
    """
    correct = (y_pred == y_true).sum().item()
    total = y_true.shape[0] * y_true.shape[1]
    return correct / total


def label_wise_accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred, axis=0)



def eval_scores(y_true, y_pred, loss, print_out=False, epoch=0, batch=0):

    f1_score = metrics.f1_score(y_pred=y_pred, y_true=y_true, average='samples', zero_division=0)
    recall = metrics.recall_score(y_pred=y_pred, y_true=y_true, average='samples', zero_division=0)
    precision = metrics.precision_score(y_pred=y_pred, y_true=y_true, average='samples', zero_division=0)
    hamming_loss = metrics.hamming_loss(y_pred=y_pred, y_true=y_true)
    ham_score = hamming_score(y_true, y_pred)
    partial_accuracy = partial_accuracy_score(y_true, y_pred)
    label_wise_accuracy = label_wise_accuracy_score(y_true, y_pred)

    if print_out:
        print_eval_scores(f1_score, recall, precision, hamming_loss, ham_score, partial_accuracy, label_wise_accuracy, loss, epoch, batch + 1)

    return (f1_score, recall, precision, hamming_loss, ham_score, partial_accuracy, label_wise_accuracy)



def print_eval_scores(f1_score, recall, precision, hamming_loss, ham_score, partial_accuracy, label_wise_accuracy, loss, epoch='Avg', batch='Avg'):
    print(f'\n======================== Epoch: {epoch}, Batch: {batch} ========================')
    print('F1 score:')
    print(f1_score)

    print('Recall:')
    print(recall)

    print('Precision:')
    print(precision)

    print('Hamming loss:')
    print(hamming_loss)

    print('Hamming score:')
    print(ham_score)

    print('Partial accuracy:')
    print(partial_accuracy)

    print('Label wise accuracy:')
    print(label_wise_accuracy)

    print('Loss:')
    print(loss)

    print('\n====================================================================\n')


