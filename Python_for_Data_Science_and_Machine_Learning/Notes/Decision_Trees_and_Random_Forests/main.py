import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


if __name__ == '__main__':
    # Read data
    df = pd.read_csv('data/kyphosis.csv')

    # Display head and basic info
    print(df.head())
    print(df.info())

    # Minimal Exploratory Data Analysis
    sns.pairplot(df, hue='Kyphosis')

    plt.show()

    # Split Features and Label
    X = df.drop('Kyphosis', axis=1)
    y = df['Kyphosis']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42
    )

    # Create Decision Tree
    dtree = DecisionTreeClassifier()

    # Train the Decision Tree
    dtree.fit(X=X_train, y=y_train)

    # Predict with the Decision Tree
    predictions = dtree.predict(X=X_test)

    # Evaluate
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    # Create Random Forest
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X=X_train, y=y_train)

    # Predict with the Random Forest
    rfc_predictions = rfc.predict(X=X_test)

    # Evaluate
    print(confusion_matrix(y_test, rfc_predictions))
    print(classification_report(y_test, rfc_predictions))
