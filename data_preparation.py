import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, precision_score


def remove_missing_values(df: pd.DataFrame, missing_value: str) -> pd.DataFrame:
    df = df.replace(missing_value, np.NaN)
    df = df.dropna()
    return df


def generete_correlation_matrix(columns):
    correlation_matrix = columns.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.savefig('output/correlationMatrix.png')


def generate_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.savefig('output/confusion_matrix.png')


def fit_and_test(classifier, X_train, y_train, X_test, y_test):
    print(classifier.__class__.__name__)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    # confusion matrix
    generate_confusion_matrix(y_test, predictions)

    # show metrics
    accurancy = accuracy_score(y_test, predictions)
    print("Accurancy:" + str(accurancy))
    recall = recall_score(y_test, predictions)
    print("Recall:" + str(recall))
    precision = precision_score(y_test, predictions)
    print("Precision:" + str(precision))


def clear_dir(dir):
    try:
        if os.path.exists(dir):
            for file in os.listdir(dir):
                file_path = os.path.join(dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Errore durante l'eliminazione del file {file_path}: {e}")
            print(f"Contenuto della cartella {dir} eliminato con successo.")
        else:
            print(f"La cartella {dir} non esiste.")
    except Exception as e:
        print(f"Errore generale durante la pulizia della cartella {dir}: {e}")