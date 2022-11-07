# breast cancer machine learning
# GridSearchCV for Models (Logistic Regression (LR) and Random Forest Classifier (RF) with Recursive Feature Elimination with Cross-validation (RFECV) - Breast cancer Wisconsin Dataset
# data attribute
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)

# Ten real-valued features are computed for each cell nucleus:
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry
# j) fractal dimension ("coastline approximation" - 1)


# Import libraries and modules and load Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
import os

# Data Description
df = pd.read_csv('data.csv')
print(df.head())

print(df.shape)
print()
df.info()
print()
print(df.describe())
print()
print(df.isnull().sum())
print()
print(df['diagnosis'].value_counts())
print()
print(df['diagnosis'].value_counts(normalize=True))
print()
print(df['diagnosis'].value_counts().plot(kind='bar'))
print(df["unnamed: 32"].value_counts())
print(df["unnamed: 32"].isna().sum())
print(df["unnamed: 32"].isna().all())
print(df["id"].isna().sum())
print(df["id"].isna().all())
print(df["diagnosis"].isna().sum())
print(df["diagnosis"].isna().all())
print(df["radius_mean"].isna().sum())

df.drop(columns="unnamed: 32", inplace=True)
print(df.shape)
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())

df.set_index('id', inplace=True)
print(df.head())
print(df.index.name)
print(df.shape)

# Exploratory Data Analysis
# missing values
print(df.isnull().sum())
print(df.isnull().sum().sum())
print(df.isnull().any().any())
print(df.isnull().any())

print(df.isnull().any().sum() == 0)
print(df.isnull().any().sum() != 0)
print(df.isnull().any().sum() > 0)
print(df.isnull().any().sum() < 0)
print(df.duplicated().sum())
print(df.duplicated().any())

# Univariate Analysis
df_x = df.drop(columns='diagnosis')
df_y = df['diagnosis']
print(df_x.head())
print(df_y.head())

Y = df.get_dummies(df_y, drop_first=True)
print(Y.head())
print(Y.shape)
print(Y.columns)
print(Y.info())
print(Y.describe())

# subgroups
mean_features = list(df_x.columns[0:10])  # mean features 0-9 index
se_features = list(df_x.columns[10:20])  # standard error
worst_features = list(df_x.columns[20:31])  # worst (or largest) mean value for mean of three largest values
print(mean_features)
print(se_features)
print(worst_features)

print(mean_features.describe())
print(se_features.describe())
print(worst_features.describe())


# dsitribution of con histograma y kde
def histograms_plot(df, col):
    plt.figure(figsize=(10, 6))
    plt.title(col)
    sns.distplot(df[col], kde=True)
    plt.show()


for col in df_x.columns:
    histograms_plot(df_x, col)

histograms_plot(mean_features, mean_features, 4, 3)
histograms_plot(se_features, se_features, 4, 3)
histograms_plot(worst_features, worst_features, 4, 3)

# standart error
histograms_plot(se_features, se_features, 4, 3)
histograms_plot(worst_features, worst_features, 4, 3)

# Countplot target
sns.countplot(df_y)
plt.show()

# Bivariate Analysis
# Correlation
corr = df_x.corr()
print(corr)

plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

count_m = df_y.value_counts()[1] / df_y.shape[0] * 100
count_b = df_y.value_counts()[0] / df_y.shape[0] * 100
print('Percentage of Malignant: {:.2f}%'.format(count_m))

# Multivariate Analysis
boxplot = df_x.boxplot(column=mean_features, grid=False, vert=False, fontsize=15, figsize=(10, 10))
plt.show()

# second method
from scipy.stats import zscore

sns.set(style="whitegrid", color_codes=True, font_scale=1.5)


def boxplot_standart(features, target):
    features_scaler = zscore(features)
    dataframe = pd.concat([target, features_scaler], axis=1)
    dataframe = pd.melt(dataframe, id_vars="diagnosis", var_name="features", value_name='value')
    fig = plt.figure(figsize=(10, 10))
    sns.boxenplot(x="features", y="value", hue="diagnosis", data=dataframe)
    plt.xticks(rotation=90)
    fig.tight_layout()
    plt.show()


boxplot_standart(mean_features, df_y)
boxplot_standart(se_features, df_y)
boxplot_standart(worst_features, df_y)

# Correlation
import seaborn as sns;

sns.set_theme()


def corr_heatmap(df, col):
    df_dummie = pd.concat([target, data], axis=1)
    corr = df_dummie.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
    matrix = np.triu(corr)
    sns.heatmap(corr, annot=True, linewidths=.5, fmt='.1f', ax=ax, cmap=cmap, mask=matrix)


corr_heatmap(df_x, df_y)


def get_positive_cor_coefficient(data, target):
    data_dummie = pd.concat([target, data], axis=1)
    positive_corr_df = data_dummie.corr(method='pearson')
    positive_corr_df = positive_corr_df.mask(np.tril(np.ones(positive_corr_df.shape)).astype(bool))
    positive_corr_df = positive_corr_df[abs(positive_corr_df) > 0.7].stack().reset_index().sort_values(by=0,
                                                                                                       ascending=False)
    positive_corr_df.rename(columns={0: "pearson_coefficient"}, inplace=True)
    return positive_corr_df


def get_negative_cor_coefficient(data, target):
    data_dummie = pd.concat([target, data], axis=1)
    negative_corr_df = data_dummie.corr(method='pearson')
    negative_corr_df = negative_corr_df.mask(np.tril(np.ones(negative_corr_df.shape)).astype(bool))
    negative_corr_df = negative_corr_df[abs(negative_corr_df) < 0.3].stack().reset_index().sort_values(by=0)
    negative_corr_df.rename(columns={0: "pearson_coefficient"}, inplace=True)
    return negative_corr_df


positive_corr_df = get_positive_cor_coefficient(df_x, df_y)
negative_corr_df = get_negative_cor_coefficient(df_x, df_y)

print(positive_corr_df)
print(negative_corr_df)

mask_corr_M = positive_corr_df["level_0"] == "M"

positive_corr_df.loc[
    positive_corr_df["level_0"] == "M"]  # Tengo las principales featurs con correlacion para que sea maligno

negative_corr = get_negative_cor_coefficient(df_x, Y)
negative_corr.head(10)

mask_corr_M = negative_corr["level_0"] == "M"
negative_corr.loc[negative_corr["level_0"] == "M"]

# Scatterplot
sns.set(style="whitegrid", color_codes=True, font_scale=1.5)
sns.pairplot(df_x, diag_kind="kde", hue="diagnosis")
plt.show()

# machine learning


# Data Procesing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
y = le.fit_transform(df_y)


# Functions for separating data
def normalize_model(target, features_select):
    """
    Separa el target y lo normaliza
    """
    y = target
    X = features_select
    scaler = preprocessing.StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X))
    return X, y


def separate_data(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

    print(f'Train Shape: {X_train.shape}')
    print(f'Test Shape: {X_test.shape}')
    return X_train, X_test, y_train, y_test


# Use RFECV
from sklearn.feature_selection import RFECV


def get_best_features_by_RFECV(model, X_train, y_train, data):
    # Use RFECV to choose the best features
    selector = RFECV(estimator=model, step=1,
                     cv=5, scoring='accuracy')
    selector = selector.fit(X_train, y_train)
    features_names = selector.get_support(1)
    best_features = data.columns[features_names]
    print(f'Optimal number of features: {selector.n_features_}')
    print()
    print(f'Best features: \n')
    print(f'{best_features}')

    selected_features = selector.get_support(1)
    X_selected_features = data[data.columns[selected_features]]
    return X_selected_features


# Area under the curve

def auc_plot(xtest, ytest, model):
    yproba = model.predict_proba(xtest)
    fpr1, tpr1, thresholds = roc_curve(ytest.astype('int'), yproba[:, 1], drop_intermediate=False)
    auc = metrics.auc(fpr1, tpr1)
    print("El AUC es = " + str(auc))

    plt.plot(fpr1, tpr1, lw=2, alpha=0.7, label='ROC curve', color='b')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Luck', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(False)
    plt.legend(loc="lower right")
    plt.title('ROC curve')
    plt.show()


# Matrix of confusion

def cm_plot(ytest, ypred):
    cm = confusion_matrix(ytest, ypred)
    df_cm = pd.DataFrame(cm, index=['Cancer', 'Control'], columns=['Cancer', 'Control'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True)
    plt.title('Classification Confusion matrix')
    plt.show()


from sklearn.metrics import accuracy_score

# I create a function to perform accuracy and also save it in a list to be able to compare it later.

list_accuracy = {}


def accuracy_model(y_test, ypred, model=""):
    test_acc = accuracy_score(y_test_selec, ypred)
    list_accuracy[model] = round(test_acc, 3)
    print("El accuracy es " + str(test_acc))


# Logic Regression with the features selected by EDA
# Select our dataset with the chosen features

features_selec = ["concave points_worst", "perimeter_worst", "concave points_mean", "radius_worst", "perimeter_mean",
                  "area_worst", "radius_mean", "area_mean"]

x_prueba = df_x.loc[:, features_selec]
x_prueba

# Split and normalize
# Ratio de 20% test y 80% train
X_train_selec, X_test_selec, y_train_selec, y_test_selec = split_and_normalize_model(y, x_prueba, 0.2, random_state=43)

# I create variable for Logistic Regression

lr = LogisticRegression()

# Fiteo data train
_ = lr.fit(X_train_selec, y_train_selec)

# I make predictions
ypred = _.predict(X_test_selec)
ypred

accuracy_model(y_test_selec, ypred, "Logistic_Regression")
auc_plot(X_test_selec, y_test_selec, lr)
cm_plot(y_test_selec, ypred)

# radnom forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf.fit(X_train_selec, y_train_selec)
ypred = rf.predict(X_test_selec)
accuracy_model(y_test_selec, ypred, "Random_Forest")
auc_plot(X_test_selec, y_test_selec, rf)
cm_plot(y_test_selec, ypred)

# logistic regression with all features
# Select our dataset with the chosen features

features_selec = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
                  "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se",
                  "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                  "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                  "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                  "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]

x_prueba = df_x.loc[:, features_selec]
x_prueba

# grip search
# I re-split and normalize only the selected features
X_train_lr_rfecv, X_test_lr_rfecv, y_train_lr_rfecv, y_test_lr_rfecv = split_and_normalize_model(y, X_select_rfecv_lr,
                                                                                                 0.2, random_state=43)

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Hyperparametros
parameters = {'penalty': ['l1', 'l2'],
              'C': [100, 10, 1.0, 0.1, 0.01],
              'solver': ['liblinear']}
#
clf_lr = GridSearchCV(lr,  # modelo
                      param_grid=parameters,  # Hyperparametros
                      refit=True,  # refit nos devuelve el modelo con los mejores parametros encontrados
                      cv=5,
                      verbose=1)  # cv indica la cantidad de folds

# Fit the model already selected
clf_lr.fit(X_train_lr_rfecv, y_train_lr_rfecv.ravel())
