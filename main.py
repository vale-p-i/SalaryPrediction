from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import *
from sklearn.model_selection import KFold
import pandas as pd
import copy

hr = "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"

# cleaning output dir
clear_dir('./output')
# uploading dataset
print("Uploading Dataset...")
dataset_path = 'input/salary.csv'
df = pd.read_csv(dataset_path)

# removing missing values
print("Removing Missing Values...")
df = remove_missing_values(df, ' ?')

# salary binary encoding
label_encoder = LabelEncoder()
df['salary'] = label_encoder.fit_transform(df['salary'])

# scaling
print(hr)
print("FEATURE SCALING")
print(hr)

# minmax normalization
minmax_scaler = MinMaxScaler()
df['age_scaled'] = minmax_scaler.fit_transform(df[['age']])
df['hours-per-week_scaled'] = minmax_scaler.fit_transform(df[['hours-per-week']])
df['capital-gain_scaled'] = minmax_scaler.fit_transform(df[['capital-gain']])
df['capital-loss_scaled'] = minmax_scaler.fit_transform(df[['capital-loss']])

# log trasformation
df['capital-gain_log'] = np.log(df['capital-gain'] + 1)
df['capital-loss_log'] = np.log(df['capital-loss'] + 1)

print("New feature created: age_scaled, hours-per-week_scaled, capital-gain_scaled, capital-loss_scaled")

# categorize age, hours-per-week, capital-gain and capital-loss
print(hr)
print("CREATING CATEGORIZED FEATURES")
print(hr)
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
    elif hours < 40:
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
distribution_comparison(df, 'age', 'age_cat', 'Age')
distribution_comparison(df, 'hours-per-week', 'hours-per-week_cat', 'HoursPerWeek')
distribution_comparison(df, 'marital-status', 'new_marital-status', 'Marital Status')
distribution_comparison(df, 'education-num', 'education', 'Education')
print("New feature created: age_cat, hours-per-week_cat, capital_cat, new_marital-status.")
print("Changed education feature")
print(hr)

# removing irrelevant data
print("REMOVING IRRELEVANT DATA")
print(hr)
df = df[df['workclass'] != ' Without-pay']
df = df[df['occupation'] != ' Armed-Forces']
print("Removed Without-pay and Armed-Forces categories")

# encoding
print(hr)
print("ENCODING")
print(hr)
print("Encoding...")
mapping = {' Before-HS': 0, ' HS-grad': 1, ' Some-college': 2, ' Assoc': 3, ' Bachelors': 4, ' Masters': 5,
           ' Doc-Prof': 6}
df['education'] = df['education'].map(mapping)

# feature selection
print(hr)
print("FEATURE SELECTION")
print(hr)
df = df.reindex(sorted(df.columns), axis=1)
num_features = [i for i in df if df[i].dtype != 'object' and df[i].dtype != 'category']
print("Generation Correlation Matrix")
generete_correlation_matrix(df[num_features])
print("Removing unselected features")
df = df.drop(columns=['native-country', 'relationship', 'fnlwgt', 'education-num', 'marital-status'])

# one hot encoding
print(hr)
print("ONE HOT ENCODING")
print(hr)
cat_features = [i for i in df if df[i].dtype == 'object' or df[i].dtype == 'category']
df_encoded = pd.get_dummies(df, columns=cat_features, dtype=int)

# Cross validation
print(hr)
print("CROSS VALIDATION")
print(hr)

y = df['salary']
X = df_encoded.drop(columns=['salary'])

kf = KFold(n_splits=10)

col_to_test = ['age_scaled', 'age_cat', 'age', 'hours-per-week', 'hours-per-week_cat', 'hours-per-week_scaled',
               'capital-loss_log', 'capital-gain_log', 'capital-gain', 'capital-loss', 'capital-gain_scaled',
               'capital-loss_scaled', 'capital_cat']

# age cat vs. age scaled
# age cat
col = copy.deepcopy(col_to_test)
features = ['age_cat', 'hours-per-week', 'capital-gain', 'capital-loss']
for i in features:
    col.remove(i)
X1 = X.drop(columns=col)
cross_validation(X1, y, features, kf, "age categorized", "cat-vs-scaledRESULTS.txt")
# age scaled
col = copy.deepcopy(col_to_test)
features = ['age_scaled', 'hours-per-week', 'capital-gain', 'capital-loss']
for i in features:
    col.remove(i)
X1 = X.drop(columns=col)
cross_validation(X1, y, features, kf, "age scaled", "cat-vs-scaledRESULTS.txt")

# hpw cat vs. hpw scaled
# hpw cat
col = copy.deepcopy(col_to_test)
features = ['age_cat', 'hours-per-week_cat', 'capital-gain', 'capital-loss']
for i in features:
    col.remove(i)
X1 = X.drop(columns=col)
cross_validation(X1, y, features, kf, "hours per week categorized", "cat-vs-scaledRESULTS.txt")

# hpw scaled
col = copy.deepcopy(col_to_test)
features = ['age_cat', 'hours-per-week_scaled', 'capital-gain', 'capital-loss']
for i in features:
    col.remove(i)
X1 = X.drop(columns=col)
cross_validation(X1, y, features, kf, "hours per week scaled", "cat-vs-scaledRESULTS.txt")

# capital gain & capital loss cat vs. scaled
# cat
col = copy.deepcopy(col_to_test)
features = ['age_cat', 'hours-per-week_cat', 'capital_cat']
for i in features:
    col.remove(i)
X1 = X.drop(columns=col)
cross_validation(X1, y, features, kf, "capital categorized", "cat-vs-scaledRESULTS.txt")

# minmax scaled
col = copy.deepcopy(col_to_test)
features = ['age_cat', 'hours-per-week_cat', 'capital-gain_scaled', 'capital-loss_scaled']
for i in features:
    col.remove(i)
X1 = X.drop(columns=col)
cross_validation(X1, y, features, kf, "Capital gain and capital log scaled", "cat-vs-scaledRESULTS.txt")

# log trasformation
col = copy.deepcopy(col_to_test)
features = ['age_cat', 'hours-per-week_cat', 'capital-gain_log', 'capital-loss_log']
for i in features:
    col.remove(i)
X1 = X.drop(columns=col)
cross_validation(X1, y, features, kf, "Capital gain log and capital loss log", "cat-vs-scaledRESULTS.txt")

# finding discriminating data
print(hr)
print("FINDING DISCRIMINATING DATA")
print(hr)
col_to_remove = ['age_scaled', 'age', 'hours-per-week', 'hours-per-week_scaled', 'capital-gain', 'capital-loss',
                 'capital-gain_scaled', 'capital-loss_scaled', 'capital_cat']
X = X.drop(columns=col_to_remove)

# testing removal of sex feature
X1 = X.drop(columns=['sex_ Female', 'sex_ Male'])
cross_validation(X1, y, X1.columns, kf, "sex", "testing-of-discriminating-featuresRESULTS.txt")

# testing removal of race feature
X1 = X.drop(columns=['race_ Asian-Pac-Islander', 'race_ Amer-Indian-Eskimo', 'race_ Black', 'race_ Other', 'race_ White'])
cross_validation(X1, y, X1.columns, kf, "race", "testing-of-discriminating-featuresRESULTS.txt")

# testing removal of age_cat feature
X1 = X.drop(columns=['age_cat'])
cross_validation(X1, y, X1.columns, kf, "age categorized", "testing-of-discriminating-featuresRESULTS.txt")

# testing removal of new_marital-status
X1 = X.drop(columns=['new_marital-status_ Divorced', 'new_marital-status_ Married', 'new_marital-status_ Never-married', 'new_marital-status_ Separated', 'new_marital-status_ Widowed'])
cross_validation(X1, y, X1.columns, kf, "marital status", "testing-of-discriminating-featuresRESULTS.txt")

# testing removal of sex and race
X1 = X.drop(columns=['sex_ Female', 'sex_ Male', 'race_ Asian-Pac-Islander', 'race_ Amer-Indian-Eskimo', 'race_ Black', 'race_ Other', 'race_ White'])
cross_validation(X1, y, X1.columns, kf, "sex and race", "testing-of-discriminating-featuresRESULTS.txt")

# testing removal of sex, race and age
X1 = X.drop(columns=['age_cat', 'sex_ Female', 'sex_ Male', 'race_ Asian-Pac-Islander', 'race_ Amer-Indian-Eskimo', 'race_ Black', 'race_ Other', 'race_ White'])
cross_validation(X1, y, X1.columns, kf, "sex, race and age", "testing-of-discriminating-featuresRESULTS.txt")

