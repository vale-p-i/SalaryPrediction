from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from data_preparation import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import copy
import random

# uploading dataset
print("Uploading Dataset...")
dataset_path = 'input/salary.csv'
df = pd.read_csv(dataset_path)

# removing missing values
print("Removing Missing Values...")
df = remove_missing_values(df, ' ?')

# scaling
print("Scaling...")
minmax_scaler = MinMaxScaler()
df['age_scaled'] = minmax_scaler.fit_transform(df[['age']])
df['hours-per-week_scaled'] = minmax_scaler.fit_transform(df[['hours-per-week']])
df['capital-gain_scaled'] = minmax_scaler.fit_transform(df[['capital-gain']])
df['capital-loss_scaled'] = minmax_scaler.fit_transform(df[['capital-loss']])

#log trasformation
df['capital-gain_log'] = np.log(df['capital-gain'] + 1)
df['capital-loss_log'] = np.log(df['capital-loss'] + 1)

# categorize age, hours-per-week, capital-gain and capital-loss
print("Categorizing...")
for i, row in df.iterrows():
    age = row['age']
    hours = row['hours-per-week']
    gain = row['capital-gain']
    loss = row['capital-loss']

    if age <= 23:
        df.loc[i, 'age_cat'] = 0
    elif 23 < age < 30:
        df.loc[i, 'age_cat'] = 1
    elif 30 <= age < 37:
        df.loc[i, 'age_cat'] = 2
    elif 37 <= age < 70:
        df.loc[i, 'age_cat'] = 3
    elif age >= 70:
        df.loc[i, 'age_cat'] = 4

    if hours == 40:
        df.loc[i, 'hours-per-week_cat'] = 1
    elif hours < 40 :
        df.loc[i, 'hours-per-week_cat'] = 0
    elif hours > 40:
        df.loc[i, 'hours-per-week_cat'] = 2

    if gain > 0 or loss > 0:
        df.loc[i, 'capital_cat'] = 1
    else:
        df.loc[i, 'capital_cat'] = 0

df = df.replace([' Assoc-voc', ' Assoc-acdm'], ' Assoc')
df = df.replace([' Preschool', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th'], ' Before-HS')
df = df.replace([' Doctorate', ' Prof-school'], ' Doc-Prof')
df['new_marital-status'] = df['marital-status']
df['new_marital-status'] = df['new_marital-status'].replace([' Married-civ-spouse', ' Married-AF-spouse'], ' Married')
df['new_marital-status'] = df['new_marital-status'].replace(' Married-spouse-absent', ' Separated')

# removing irrelevant data
df = df[df['workclass'] != ' Without-pay']
df = df[df['occupation'] != ' Armed-Forces']

# encoding
print("Encoding...")
mapping = {' Before-HS': 0, ' HS-grad': 1, ' Some-college': 2, ' Assoc':3, ' Bachelors':4, ' Masters':5, ' Doc-Prof':6}
df['education'] = df['education'].map(mapping)
label_encoder = LabelEncoder()
df['salary'] = label_encoder.fit_transform(df['salary'])

num_features = [i for i in df if df[i].dtype != 'object' and df[i].dtype != 'category']
print(num_features)

generete_correlation_matrix(df[num_features])

df = df.drop(columns=['native-country', 'relationship', 'fnlwgt', 'education-num', 'marital-status', 'age'])

cat_features = [i for i in df if df[i].dtype == 'object' or df[i].dtype == 'category']
print(cat_features)
df_encoded = pd.get_dummies(df, columns=cat_features, dtype=int)

y = df['salary']
X = df_encoded.drop(columns=['salary'])

# train and test
col_to_test = ['age_scaled', 'age_cat', 'hours-per-week_cat', 'hours-per-week_scaled', 'capital-loss_log', 'capital-gain_log', 'capital-gain_scaled', 'capital-loss_scaled', 'capital_cat']
col = copy.deepcopy(col_to_test)
for i in ['age_cat', 'hours-per-week_cat', 'capital_cat']:
  col.remove(i)
X1 = X.drop(columns=col)
x = random.randint(0, 256)
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=x)
fit_and_test(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)
