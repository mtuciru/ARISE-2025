import torch
from sklearn.metrics import accuracy_score, f1_score


def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)

    return f1_score(targets.cpu(), preds.cpu(), average='micro')