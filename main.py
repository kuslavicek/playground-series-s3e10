# import autoviz
# import ydata_profiling
import os
import pandas as pd
# import sweetviz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import dtale
import numpy as np


def eda(csvfile):
    if not os.path.exists("reports"):
        os.mkdir("reports")
    # profile = ydata_profiling.ProfileReport(pd.read_csv(csvfile), explorative=True)
    # profile.to_file("reports/profile.html")
    # av = autoviz.AutoViz_Class()
    # auto = av.AutoViz(filename=csvfile, chart_format='html', save_plot_dir="reports")
    # sv = sweetviz.analyze(pd.read_csv(csvfile))
    dtale.show(pd.read_csv(csvfile), subprocess=False)
    return 0


def train_logistic(x, y):
    log_regressor = LogisticRegression(C=0.7, max_iter=400)
    scores = cross_validate(log_regressor, x, y, scoring="neg_log_loss", verbose=1, cv=5, return_train_score=True,
                            return_estimator=True)
    print("Logistic train: " + str(scores['train_score']))
    print("Logistic test: " + str(scores['test_score']))
    log_regressor = scores['estimator'][2]
    predictions = log_regressor.predict_proba(x)
    print("Logloss: " + str(log_loss(y, predictions)))
    return 0


def train_neighbors(x, y):
    neighbors= KNeighborsClassifier()
    scores = cross_validate(neighbors, x, y, scoring="neg_log_loss", verbose=1, cv=5, return_train_score=True,
                              return_estimator=True)
    print("Neighbors train: " + str(scores['train_score']))
    print("Neighbors test: " + str(scores['test_score']))
    return 0


def train_decision(x, y):
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3)
    scores = cross_validate(tree, x, y, scoring="neg_log_loss", verbose=1, cv=5, return_train_score=True,
                            return_estimator=True)
    print("Tree train: " + str(scores['train_score']))
    print("Tree test: " + str(scores['test_score']))
    tree = scores['estimator'][2]
    predictions = tree.predict_proba(x)
    print("Logloss: "+str(log_loss(y, predictions)))
    return 0


def preprocess(csvfile):
    df = pd.read_csv(csvfile)
    y = np.asarray(df['Class'])
    df = df.drop(['Class'], axis=1)
    df = df.drop(['id'], axis=1)
    df = df.drop(['Skewness_DMSNR_Curve'], axis=1)
    scaler = StandardScaler()
    x = scaler.fit_transform(df)
    train_logistic(x, y)
    # train_neighbors(x,y)
    train_decision(x,y)
    return 0


if __name__ == '__main__':
    # eda("train.csv")
    preprocess("train.csv")
