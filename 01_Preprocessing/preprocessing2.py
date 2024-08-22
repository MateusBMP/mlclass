import matplotlib.pyplot as plt
import pandas as pd
import random
import requests
import sys

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

VERBOSE_MODE = True

DATASET = 'diabetes_dataset.csv'
PREDICT_SET = 'diabetes_app.csv'
FEATURE_COLS = ['Pregnancies', 'Glucose', 'Insulin', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']
TARGET_COL = 'Outcome'

def send_predictions(predictions: pd.Series) -> str:
    # Enviando previsões realizadas com o modelo para o servidor
    URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"
    DEV_KEY = "Mexerica"

    # json para ser enviado para o servidor
    data = {'dev_key':DEV_KEY,
            'predictions':predictions.to_json(orient='values')}

    # Enviando requisição e salvando o objeto resposta
    r = requests.post(url = URL, data = data)

    # Extraindo e imprimindo o texto da resposta
    return r.text

def create_knn_model(X: pd.DataFrame, y: pd.Series) -> KNeighborsClassifier:
    neigh = KNeighborsClassifier(n_neighbors=3)
    return neigh.fit(X, y)

def create_linear_regression_model(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    lr = LinearRegression()
    return lr.fit(X, y)

def predict_knn(model: KNeighborsClassifier, data: pd.DataFrame) -> pd.Series:
    return model.predict(data)

def _calculate_insulin_value(row: pd.Series, model: LinearRegression, rmn_min: float, rmn_max: float, poly: PolynomialFeatures|None = None) -> float:
    if not pd.isnull(row['Insulin']):
        return row['Insulin']
    
    # Calculate a relative value between rmn_min and rmn_max equivalent to the Glucose value
    rmn = (rmn_max - rmn_min) * row['Glucose'] + rmn_min

    # Sort a value between -rmn and rmn rounding to 6 decimal places
    rmn = random.randint(round(-rmn * 10**6), round(rmn * 10**6)) / 10**6

    # Predict the Insulin value using the Linear Regression model
    X_data = pd.DataFrame(pd.DataFrame({'Glucose': [row['Glucose']]}))
    if poly is not None:
        X_data = poly.transform(X_data)
    prediction = model.predict(X_data)[0]

    # Sum the predicted value by the relative value
    return prediction + rmn

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Select only the feature columns and the target column
    df = df[[TARGET_COL] + FEATURE_COLS]

    # Normalize the dataframe
    df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

    # Remove rows without Glucose column values
    df = df[df['Glucose'].notnull()]

    # Infer missing Insulin values using Linear Regression with the Glucose column
    # df_to_train = df[df['Glucose'].notnull() & df['Insulin'].notnull()]
    # model = create_linear_regression_model(X = df_to_train[['Glucose']], y = df_to_train['Insulin'])
    # df['Insulin'] = df.apply(lambda row: _calculate_insulin_value(row, model, rmn_min=0.01, rmn_max=0.2), axis=1)

    # Infer missing Insulin values using Polynomial Regression with the Glucose column
    # df_to_train = df[df['Glucose'].notnull() & df['Insulin'].notnull()]
    # poly = PolynomialFeatures(degree=2)
    # model = create_linear_regression_model(X = poly.fit_transform(df_to_train[['Glucose']]), y = df_to_train['Insulin'])
    # df['Insulin'] = df.apply(lambda row: _calculate_insulin_value(row, model, rmn_min=0.01, rmn_max=0.2, poly = poly), axis=1)

    # Remove rows with missing values   
    df = df.dropna()

    # Remove outliers between 0.05 and 0.95 quantiles
    # for col in FEATURE_COLS:
    #     lower_bound = df[col].quantile(0.05)
    #     upper_bound = df[col].quantile(0.95)
    #     df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]

    return df

def run(algorithm: str, send: bool) -> None:
    if algorithm not in ['knn']:
        print(f'Invalid algorithm: {algorithm}')
        print('Valid algorithms: knn')
        sys.exit(1)

    # Load the dataset
    df = pd.read_csv(DATASET)

    # Preprocess the dataset
    df = preprocess(df)

    # Create the KNN model
    model = create_knn_model(df[FEATURE_COLS], df.get(TARGET_COL))

    # Predict the target values
    predict_df = pd.read_csv(PREDICT_SET)
    predict_df = pd.DataFrame(MinMaxScaler().fit_transform(predict_df), columns=predict_df.columns) # Normalize the predict dataset
    predictions = pd.Series(predict_knn(model, predict_df[FEATURE_COLS]))

    if VERBOSE_MODE:
        print(predictions.describe())
        print(predictions.values)

    # Send the predictions
    if send:
        response = send_predictions(predictions)
        if VERBOSE_MODE:
            print('Response:')
            print(response)

def analyze(apply_preprocess: bool) -> None:
    # Load the dataset
    df = pd.read_csv(DATASET)

    # Preprocess the dataset
    if apply_preprocess:
        df = preprocess(df)

    df.info()
    print(df.describe())
    print(df.corr())

    # Plot the boxplot
    df.boxplot()
    plt.show()

    # Plot the scatter matrix
    colors = {0: 'blue', 1: 'red'} # Filter into two groups: Outcome = 0 and Outcome = 1
    df['Color'] = df['Outcome'].apply(lambda x: colors[x])
    pd.plotting.scatter_matrix(df, alpha = 0.8, c=df['Color'], diagonal='kde')
    plt.show()

    # Plot the frequency chart between the Outcome and the other columns
    fig, axs = plt.subplots(1, len(FEATURE_COLS))
    groups = df.groupby('Outcome')
    for i, col in enumerate(FEATURE_COLS):
        for name, group in groups:
            group[col].value_counts().sort_index().plot(kind='bar', ax=axs[i], color=colors[name], alpha=0.8)
        axs[i].set_title(col)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ['run:knn', 'analyze'] or sys.argv[1] == 'help':
        print('Usage: python preprocessing2.py <command>')
        print('Commands:')
        print('  run:knn [send]        Run the KNN model. Use "send" to send the predictions')
        print('  analyze [preprocess]  Analyze the dataset and show the results. Use "preprocess" to preprocessed dataset before analyzing')
        sys.exit(1)

    if sys.argv[1] == 'run:knn':
        run(algorithm = 'knn', send = (len(sys.argv) > 2 and sys.argv[2] == 'send'))
    elif sys.argv[1] == 'analyze':
        analyze(apply_preprocess = (len(sys.argv) > 2 and sys.argv[2] == 'preprocess'))
