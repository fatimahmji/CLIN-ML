import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
)


def metrics(labels, probabilities):
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        labels, probabilities >= 0.5, labels=[0, 1]
    ).ravel()
    return dict(
        auroc=float(roc_auc_score(labels, probabilities)),
        auprc=float(average_precision_score(labels, probabilities)),
        sensitivity=float(true_positives / (true_positives + false_negatives)),
        specificity=float(true_negatives / (true_negatives + false_positives)),
        f1=float(f1_score(labels, probabilities >= 0.5)),
        brier=float(brier_score_loss(labels, probabilities)),
        confusion_matrix=[
            [int(true_negatives), int(false_positives)],
            [int(false_negatives), int(true_positives)],
        ],
    )
