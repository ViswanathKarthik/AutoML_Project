"""
Plot utilities for AutoML report
"""

import matplotlib.pyplot as plt
import tempfile


def create_histogram(series):
    """
    Create histogram for a feature and return image path
    """

    fig = plt.figure()

    series.hist(bins=30)

    plt.title(series.name)
    plt.xlabel(series.name)
    plt.ylabel("Frequency")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    plt.savefig(tmp.name)

    plt.close()

    return tmp.name


def create_roc_plot(y_test, y_prob):
    """
    Generate ROC curve image
    """

    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()

    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    plt.savefig(tmp.name)

    plt.close()

    return tmp.name


def create_confusion_matrix(y_test, y_pred):
    """
    Generate confusion matrix image
    """

    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    plt.savefig(tmp.name)

    plt.close()

    return tmp.name