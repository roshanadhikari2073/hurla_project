
def evaluate(preds, labels):
    """
    Evaluate classification performance using standard metrics without
    external dependencies. Handles cases where only a single class is
    present in ``preds`` or ``labels``.
    """
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

    total = len(labels)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "Accuracy": accuracy,
        "F1": f1,
        "FPR": fpr,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }

