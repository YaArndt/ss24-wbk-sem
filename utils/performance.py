# Description:
# Class to evaluate model performance using different metrics.

# =================================================================================================

import functools
import torch
from torch import Tensor

# =================================================================================================

def validate_inputs(func):
    @functools.wraps(func)
    def wrapper(self, predictions, labels, *args, **kwargs):
        if not torch.is_tensor(predictions):
            raise ValueError("Predictions must be a torch.Tensor")
        if not torch.is_tensor(labels):
            raise ValueError("Labels must be a torch.Tensor")
        if predictions.shape != labels.shape:
            raise ValueError("Predictions and labels must have the same shape")
        return func(self, predictions, labels, *args, **kwargs)
    return wrapper


class Performance:
    """Class to evaluate model performance using different metrics.
    """
    def __init__(self, pos_label, neg_label) -> None:
        """Perofmrance class constructor to initialize the positive and negative labels.

        Args:
            pos_label (any): Positive label in the dataset
            neg_label (any): Negative label in the dataset
        """
        self.pos_label = pos_label
        self.neg_label = neg_label

    def cfm(self, predictions: Tensor, labels: Tensor) -> dict:
        """Calculate the confusion matrix for the given predictions and labels.

        Args:
            predictions (Tensor): Predictions from the model
            labels (Tensor): Labels from the dataset

        Returns:
            dict: TP, TN, FP, FN in the confusion matrix in a dictionary
        """
        tp = torch.sum((predictions == self.pos_label) & (labels == self.pos_label))
        tn = torch.sum((predictions == self.neg_label) & (labels == self.neg_label))
        fp = torch.sum((predictions == self.pos_label) & (labels == self.neg_label))
        fn = torch.sum((predictions == self.neg_label) & (labels == self.pos_label))

        cf = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }

        return cf
    
    @validate_inputs
    def accuracy(self, predictions: Tensor, labels: Tensor) -> float:
        """Calculate the accuracy of the model.

        Args:
            predictions (Tensor): Model predictions
            labels (Tensor): Data labels

        Returns:
            float: Accuracy of the model
        """
        cf = self.cfm(predictions, labels)
        acc = (cf["TP"] + cf["TN"]) / (cf["TP"] + cf["TN"] + cf["FP"] + cf["FN"])
        return acc
    
    @validate_inputs
    def precision(self, predictions: Tensor, labels: Tensor) -> float:
        """Calculate the precision of the model.

        Args:
            predictions (Tensor): Model predictions
            labels (Tensor): Data labels

        Returns:
            float: Precision of the model
        """
        cf = self.cfm(predictions, labels)
        prec = cf["TP"] / (cf["TP"] + cf["FP"])
        return prec
    
    @validate_inputs
    def recall(self, predictions: Tensor, labels: Tensor) -> float:
        """Calculate the recall of the model.

        Args:
            predictions (Tensor): Model predictions
            labels (Tensor): Data labels

        Returns:
            float: Recall of the model
        """
        cf = self.cfm(predictions, labels)
        rec = cf["TP"] / (cf["TP"] + cf["FN"])
        return rec
    
    @validate_inputs
    def f1_score(self, predictions: Tensor, labels: Tensor) -> float:
        """Calculate the F1-Score of the model.

        Args:
            predictions (Tensor): Model predictions
            labels (Tensor): Data labels

        Returns:
            float: F1-Score of the model
        """
        precision = self.precision(predictions, labels)
        recall = self.recall(predictions, labels)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
