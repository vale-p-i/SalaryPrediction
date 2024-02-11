from data_preparation import *
import pandas as pd

# uploading dataset
dataset_path = 'input/salary.csv'
df = pd.read_csv(dataset_path)

# removing missing values
df = remove_missing_values(df, ' ?')
