import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':
    cancer = load_breast_cancer()

    # Show the description of the Cancer Dataset
    print(cancer['DESCR'], end='\n\n')

    # Save Features Data to DataFrame
    df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

    # Show DataFrame Info
    print(df_feat.info(), end='\n\n')

    # Show DataFrame Description
    print(df_feat.describe(), end='\n\n')

    # Show Target Set
    print(cancer['target'], end='\n\n')

    # Show Target Set Categories
    print(cancer['target_names'], end='\n\n')

    X = df_feat
    y = cancer['target']

    # Split the Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    model = SVC()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(confusion_matrix(y_test, predictions), end='\n\n')

    print(classification_report(y_test, predictions), end='\n\n')

    param_grid = {
        'C': [0.1, 1, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.00001]
    }

    grid = GridSearchCV(SVC(), param_grid=param_grid, verbose=10)

    grid.fit(X_train, y_train)

    print(grid.best_params_, end='\n\n')

    print(grid.best_estimator_, end='\n\n')

    grid_predictions = grid.best_estimator_.predict(X_test)

    print(confusion_matrix(y_test, grid_predictions), end='\n\n')

    print(classification_report(y_test, grid_predictions), end='\n\n')
