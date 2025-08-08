import sys
from pathlib import Path

# Ensure project root is on the Python path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.evaluation import evaluate


def test_evaluate_all_negative():
    preds = [0, 0, 0]
    labels = [0, 0, 0]
    metrics = evaluate(preds, labels)
    assert metrics["TP"] == 0
    assert metrics["FP"] == 0
    assert metrics["TN"] == 3
    assert metrics["FN"] == 0
    assert metrics["Accuracy"] == 1
    assert metrics["F1"] == 0
    assert metrics["FPR"] == 0


def test_evaluate_all_positive():
    preds = [1, 1, 1]
    labels = [1, 1, 1]
    metrics = evaluate(preds, labels)
    assert metrics["TP"] == 3
    assert metrics["FP"] == 0
    assert metrics["TN"] == 0
    assert metrics["FN"] == 0
    assert metrics["Accuracy"] == 1
    assert metrics["F1"] == 1
    assert metrics["FPR"] == 0

