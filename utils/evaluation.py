from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def evaluate(preds, labels):
    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # Standard metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Return extended metrics for deeper reward shaping
    return {
        "Accuracy": acc,
        "F1": f1,
        "FPR": fpr,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }