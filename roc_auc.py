import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

def run_eval():
    y_true = pd.read_csv('./results/gt.txt', index_col=0).to_numpy()
    y_pred = pd.read_csv('./results/pred.txt', index_col=0).to_numpy()

    # Example ground truth (true labels) and predicted probabilities for multilabel classification
    # Each row is a sample, and each column is a class (multilabel)
    # y_true = np.array([[1, 0, 1],  # Sample 1: Class 1 and Class 3
    #                    [0, 1, 0],  # Sample 2: Class 2
    #                    [1, 1, 1],  # Sample 3: Class 1, Class 2, Class 3
    #                    [0, 0, 0],  # Sample 4: No class
    #                    [1, 0, 0],  # Sample 5: Class 1
    #                    [0, 1, 1]])  # Sample 6: Class 2 and Class 3

    # # Example predicted probabilities (model's output probabilities for each class)
    # y_pred = np.array([[0.8, 0.1, 0.9],  # High probability for class 1 and class 3
    #                    [0.1, 0.7, 0.2],  # High probability for class 2
    #                    [0.7, 0.8, 0.9],  # High probability for class 1, class 2, and class 3
    #                    [0.3, 0.2, 0.1],  # Low probabilities for all classes
    #                    [0.9, 0.1, 0.2],  # High probability for class 1
    #                    [0.2, 0.8, 0.7]])  # High probability for class 2 and class 3

    # Number of classes
    n_classes = y_true.shape[1]

    # Store ROC curve information for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds_dict = dict()

    # Calculate ROC curve and AUC for each label (class)
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        thresholds_dict[i] = thresholds

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 6))

    # Plot each class's ROC curve
    for i in range(n_classes):
        if i != 6 and i != 9:
            continue
        plt.plot(fpr[i], tpr[i], color='tab:blue', lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        step = len(thresholds_dict[i]) // 5  # Choose to show 5 threshold points
        selected_thresholds = thresholds_dict[i][::step]

        for j, threshold in enumerate(selected_thresholds):
            # Find the corresponding TPR and FPR for the selected threshold
            idx = np.where(thresholds_dict[i] == threshold)[0][0]
            plt.scatter(fpr[i][idx], tpr[i][idx], color='red')  # Red dots to indicate selected thresholds

            # Annotate thresholds on the curve
            plt.annotate(f'{threshold:.2f}', (fpr[i][idx], tpr[i][idx]), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8)



    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

    # Add labels and title
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Multilabel Classification')
    plt.legend(loc='lower right')

    # Show the plot
    plt.show()

    # Display thresholds for each class
    for i in range(n_classes):
        print(f"Class {i} thresholds: {thresholds}")
