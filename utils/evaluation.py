from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def evaluate(preds, labels):
    """
    Evaluate classification performance using standard metrics,
    including full confusion matrix components for reward shaping.
    """

    # Compute confusion matrix: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # Calculate standard classification metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Return a dictionary of evaluation metrics with clarity
    return {
        "Accuracy": accuracy,
        "F1": f1,
        "FPR": fpr,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }