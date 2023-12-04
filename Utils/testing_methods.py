from sklearn.metrics import precision_score, mean_squared_error, accuracy_score, recall_score, f1_score
import numpy as np


def classifier_score(predict, true, *args):
    y_pred = predict.argmax(axis=1)
  #  print(y_pred.shape,true.shape)
    prec = []
    acc = []
    rec = []
    f1 = []
    for i in range(true.shape[1]):
        prec.append(precision_score(y_pred=y_pred[:, i],
                                    y_true=true[:, i], average='micro'))
        acc.append(accuracy_score(y_pred=y_pred[:, i],
                                  y_true=true[:, i]))
        rec.append(recall_score(y_pred=y_pred[:, i],
                                y_true=true[:, i], average='micro'))
        f1.append(f1_score(y_pred=y_pred[:, i],
                           y_true=true[:, i], average='micro'))
    return np.median(f1)


def check_model_classifier(score_val):
    if score_val > 0.95:
        return True
    else:
        return False
