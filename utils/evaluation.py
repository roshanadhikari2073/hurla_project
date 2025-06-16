from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def evaluate(preds, labels):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    fpr = fp / (fp + tn)
    return {"Accuracy": acc, "F1": f1, "FPR": fpr}
