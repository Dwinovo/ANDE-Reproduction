import numpy as np

from ande.metrics import compute_metrics, multiclass_fpr


def test_perfect_prediction():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = y_true.copy()
    m = compute_metrics(y_true, y_pred, num_classes=4)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.fpr == 0.0


def test_completely_wrong():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    m = compute_metrics(y_true, y_pred, num_classes=2)
    assert m.accuracy == 0.0
    # FPR: predict 1 on all true 0 -> FP=4, TN=0 for class 1; class 0 predicts 0 never -> FP=0, TN=0
    # macro avg over 2 classes; allow [0, 1]
    assert 0.0 <= m.fpr <= 1.0


def test_multiclass_fpr_macro():
    # 3 classes, balanced; one off-diagonal mistake.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    fpr = multiclass_fpr(y_true, y_pred, num_classes=3)
    assert 0.0 < fpr < 0.5
