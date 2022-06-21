from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from sklearn.metrics import roc_curve, precision_score, average_precision_score, precision_recall_curve
import numpy as np


def data_transform(data):
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise ValueError("data style must be list or array")
    return data


def get_auc_score(y_true, y_score):
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)
    try:
        score = roc_auc_score(y_true=y_true, y_score=y_score,)
    except:
        score = np.nan
    return score


def get_accuracy_score(y_true, y_pred):
    y_true = data_transform(y_true)
    y_pred = data_transform(y_pred)
    score = accuracy_score(y_true=y_true, y_pred=y_pred)
    return score


def get_f1_score(y_true, y_pred):
    y_true = data_transform(y_true)
    y_pred = data_transform(y_pred)
    score = f1_score(y_true=y_true, y_pred=y_pred)
    return score


def get_recall_score(y_true, y_pred):
    y_true = data_transform(y_true)
    y_pred = data_transform(y_pred)
    score = recall_score(y_true=y_true, y_pred=y_pred)
    return score


def get_precision_score(y_true, y_pred):
    y_true = data_transform(y_true)
    y_pred = data_transform(y_pred)
    score = precision_score(y_true=y_true, y_pred=y_pred)
    return score


def get_average_precision_score(y_true, y_score):
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)
    score = average_precision_score(y_true=y_true, y_score=y_score)
    return score


def get_roc_best_threshold(y_true, y_score, pos_label=None):
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)

    # sensitivity = tpr
    # 1-specificity = fpr
    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=pos_label)  # 计算真正率和假正率
    RightIndex = tpr + (1-fpr) - 1
    index = np.argmax(RightIndex)
    # RightIndex_val = RightIndex[index]
    # tpr_val = tpr[index]
    # fpr_val = fpr[index]
    threshold_val = threshold[index]
    return threshold_val


def get_mAP_best_threshold(y_true, y_score, pos_label=None):
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)

    precision, recall, threshold = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    RightIndex = precision + recall
    index = np.argmax(RightIndex)
    threshold_val = threshold[index]
    return threshold_val


def get_all_metrics(positive_prob, outcome, label):
    acc = get_accuracy_score(y_true=label,
                             y_pred=outcome)
    precision = get_precision_score(y_true=label,
                                    y_pred=outcome)
    recall = get_recall_score(y_true=label,
                              y_pred=outcome)
    f1_score = get_f1_score(y_true=label,
                            y_pred=outcome)
    auc = get_auc_score(y_true=label,
                        y_score=positive_prob)
    average_precision = get_average_precision_score(y_true=label,
                                                    y_score=positive_prob)
    best_roc_auc_threshold = get_roc_best_threshold(y_true=label,
                                                    y_score=positive_prob)
    best_mAP_auc_threshold = get_mAP_best_threshold(y_true=label,
                                                    y_score=positive_prob)
    return acc, precision, recall, f1_score, auc, average_precision, best_roc_auc_threshold, best_mAP_auc_threshold