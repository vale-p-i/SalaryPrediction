import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import statistics
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

hr = '----------------------------------------------------------------------------------------------------'


def remove_missing_values(df: pd.DataFrame, missing_value: str) -> pd.DataFrame:
    df = df.replace(missing_value, np.NaN)
    df = df.dropna()
    return df


def generate_correlation_matrix(columns):
    correlation_matrix = columns.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.savefig('output/correlationMatrix.png')


def cross_validation(X, y, features, validator, feature_tested, filename):
    report = open('output/'+filename, 'a')
    tests = list(["accuracy", "precision", "recall"])
    cv_score = cross_validate(DecisionTreeClassifier(), X, y, cv=validator, n_jobs=3, verbose=0, scoring=tests)
    accuracy_mean = statistics.mean(cv_score['test_accuracy'])
    precision_mean = statistics.mean(cv_score['test_precision'])
    recall_mean = statistics.mean(cv_score['test_recall'])
    report.write("\n")
    report.write("TESTING " + feature_tested + "\n")
    report.write("Using features: " + str(features) + "\n")
    report.write("Precision: " + str(precision_mean) + "\n")
    report.write("Recall: " + str(recall_mean) + "\n")
    report.write("Accuracy: " + str(accuracy_mean) + "\n")
    report.write("\n")
    report.write(hr + "\n")


def clear_dir(directory):
    try:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed deleting {file_path}: {e}")
            print(f"{directory} elements successfully deleted")
        else:
            print(f"{directory} does not exist")
    except Exception as e:
        print(f"Failed cleaning {directory}: {e}")


def distribution_comparison(df: pd.DataFrame, colname_1, colname_2, varname):
    fig, ax = plt.subplots(2, 1, figsize=(20, 16))
    sns.countplot(x=colname_1, hue='salary', data=df, palette='Set2', ax=ax[0])
    ax[0].set_title(f'Before ( {varname} Distribution)')
    sns.countplot(x=colname_2, hue='salary', data=df, palette='Set2', ax=ax[1])
    ax[1].set_title(f'After ( {varname} Distribution)')
    plt.savefig(f'output/{varname}DistributionComparison.png')
    plt.close()
