import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from . import _eval_protocols as eval_protocols
from utils.data import load_UCR


def eval_classification(model, model_name, dataset_name, dim_series, eval_protocol="linear"):
    train_data, train_labels, test_data, test_labels = load_UCR(dataset_name)
    train_data = train_data.reshape([-1, 1, dim_series])
    test_data = test_data.reshape([-1, 1, dim_series])

    if model_name == 'seanet':
        train_embedding = model.encode(torch.from_numpy(train_data).to('cuda'))
        test_embedding = model.encode(torch.from_numpy(test_data).to('cuda'))
    elif model_name == 'tsrnet':
        _, _, _, train_embedding = model.forward(
            torch.from_numpy(train_data).to('cuda'))
        _, _, _, test_embedding = model.forward(
            torch.from_numpy(test_data).to('cuda'))

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_embedding = merge_dim01(train_embedding)
        train_labels = merge_dim01(train_labels)
        test_embedding = merge_dim01(test_embedding)
        test_labels = merge_dim01(test_labels)

    train_embedding = train_embedding.detach().cpu().numpy()
    test_embedding = test_embedding.detach().cpu().numpy()

    clf = fit_clf(train_embedding, train_labels)

    acc = clf.score(test_embedding, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_embedding)
        y_score = np.argmax(y_score, axis=1)

    test_labels_onehot = label_binarize(
        test_labels, classes=np.arange(train_labels.max() + 1))
    y_score_onehot = label_binarize(
        y_score, classes=np.arange(train_labels.max() + 1))
    auprc = average_precision_score(test_labels_onehot, y_score_onehot)

    # return y_score, {'acc': acc, 'auprc': auprc}
    return y_score, acc, auprc
