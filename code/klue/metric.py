import numpy as np
import sklearn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


# TODO : ADD TYPE HINT!
def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        'competition:year',
        'competition:place',
        'org:country_of_headquarters',
        'country:number_of_wins',
        'parent relationship:sub relationship',
        'country:population',
        'country:faced_country',
        'org:alternative_name',
        'no_relation',
        'country:soccer_player',
        'country:game_results',
        'country:FIFA_ranking',
        'country:year_of_victory'
        ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return (
        sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices)
        * 100.0
    )


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(13)[labels]

    score = np.zeros((13,))
    for c in range(13):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c
        )
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }
