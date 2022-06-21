import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
from utils.nn_metrics import data_transform


def plot_training_metrics(history, save_dir=None, plot_type="loss", title=None, is_show=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if plot_type not in ["loss", "acc", "roc_auc", "average_precision"]:
        raise ValueError("plot_type error! this should be 'loss' 'accuracy' 'roc_auc' or 'average_precision'")
    train_plot_value = history[plot_type]
    val_plot_value = history["val_"+plot_type]
    epochs = range(1,len(train_plot_value)+1)

    plt.figure()
    plt.plot(epochs, train_plot_value, "r-", label="train")
    plt.plot(epochs, val_plot_value, "b--", label="val")
    plt.legend()
    if title is None:
        title = f"train and validation {plot_type}"
    plt.title(title)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
    if is_show:
        plt.show()


def plot_training_data(data, save_dir=None, plot_type="loss", title=None, is_show=False):
    epochs = range(1, len(data) + 1)
    plt.figure()
    plt.plot(epochs, data, "b-", label="val")
    plt.legend()
    if title is None:
        title = f"train and validation {plot_type}"
    plt.title(title)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
    if is_show:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, labels, title=None, save_dir=None, is_percentage=False, is_show=False):
    y_true = data_transform(y_true)
    y_pred = data_transform(y_pred)
    if title is None:
        title = "confusion matrix"

    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    if is_percentage:
        matrix = (matrix.T/np.sum(matrix, axis=1)).T
        fmt = ".4g"
    else:
        fmt = ".20g"
    plt.figure()
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=ax, fmt=fmt)
    ax.set_title(title)
    ax.set_xlabel("predict")
    ax.set_ylabel("true")

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
    if is_show:
        plt.show()
    plt.close()
    sns.reset_defaults()


def save_history_to_csv(history, output_dir, filename):
    fconv = open(os.path.join(output_dir, filename), 'w')
    fconv.write(' ,Train, , , , , , ,Validation, , , , , , ,\n')
    fconv.write("epoch,train_loss,train_acc,train_precision,train_recall,train_f1,train_auc_roc,train_average_precision,val_loss,val_acc,val_precision,val_recall,val_f1,val_auc_roc,val_average_precision")
    for i in range(len(history["loss"])):
        result = "\n" + str(i+1) + ',' + str(round(history["loss"][i], 4)) + ',' + str(round(history["acc"][i], 4)) + ',' \
                 + str(round(history["precision"][i], 4)) + ',' + str(round(history["recall"][i], 4)) + ',' \
                 + str(round(history["f1_score"][i], 4)) + ',' + str(round(history["roc_auc"][i], 4)) + ',' \
                 + str(round(history["average_precision"][i], 4)) + ','\
                 + str(round(history["val_loss"][i], 4)) + ',' + str(round(history["val_acc"][i], 4)) + ',' \
                 + str(round(history["val_precision"][i], 4)) + ',' + str(round(history["val_recall"][i], 4)) \
                 + ',' + str(round(history["val_f1_score"][i], 4)) + ',' + str(round(history["val_roc_auc"][i], 4)) \
                 + ',' + str(round(history["val_average_precision"][i], 4))
        fconv.write(result)
    fconv.close()




def plot_roc_auc(y_true, y_score, pos_label=None, tiltle_suffix="", save_dir=None, is_show=False):
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)
    # Compute ROC curve and ROC area for each class

    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=pos_label)  # 计算真正率和假正率
    roc_auc = auc(fpr, tpr)  # 计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic'+tiltle_suffix)
    plt.legend(loc="lower right")

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "Receiver operating characteristic{}.png".format(tiltle_suffix)))
    if is_show:
        plt.show()
    plt.close()


def plot_precision_recall(y_true, y_score, pos_label=None, save_dir=None, is_show=False):
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)
    # Compute PR curve and PR area for each class
    precision, recall, threshold = precision_recall_curve(y_true, y_score, pos_label=pos_label)  # 计算精确率和召回率
    pr_score = average_precision_score(y_true=y_true, y_score=y_score)  # 计算预测值的平均准确率,该分数对应于presicion-recall曲线下的面积

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision,color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % pr_score)  # 召回率为横坐标，精确率为纵坐标做曲线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "Precision-Recall.png"))
    if is_show:
        plt.show()
    plt.close()