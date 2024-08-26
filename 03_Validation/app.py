import sys
import requests

from pandas import DataFrame, Series, option_context, read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def send_predictions(predictions: Series) -> str:
    # Enviando previsões realizadas com o modelo para o servidor
    URL = "https://aydanomachado.com/mlclass/03_Validation.php"
    DEV_KEY = "Mexerica"

    # json para ser enviado para o servidor
    data = {'dev_key':DEV_KEY,
            'predictions':predictions.to_json(orient='values')}

    # Enviando requisição e salvando o objeto resposta
    r = requests.post(url = URL, data = data)

    # Extraindo e imprimindo o texto da resposta
    return r.text

class Knn:
    def preprocess(self, df: DataFrame) -> tuple[DataFrame, Series]:
        y = df.pop('type')

        # Convert the sex column to integers
        df = self.convert_sex_to_int(df)

        # Normalize the dataset
        X = DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

        return X, y
    
    def train(self, X: DataFrame, y: Series) -> None:
        neigh = KNeighborsClassifier(n_neighbors=3)
        return neigh.fit(X, y)
    
    def predict(self, model: KNeighborsClassifier, df: DataFrame) -> Series:
        return Series(model.predict(df))
    
    def convert_sex_to_int(self, df: DataFrame) -> DataFrame:
        with option_context('future.no_silent_downcasting', True):
            df['sex'] = df['sex'].replace({'M': 0, 'F': 1, 'I': 2})
        return df

class Analysis:
    def show_info(self, df: DataFrame) -> None:
        df.info()

    def show_description(self, df: DataFrame, normalize: bool = False) -> None:
        if normalize:
            sex = df.pop('sex')
            type = df.pop('type')
            df = DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
            df['sex'] = sex
            df['type'] = type

        print(df.describe())

    def show_correlation(self, df: DataFrame) -> None:
        df = df.drop('sex', axis=1)
        print(df.corr())

class Menu:
    def __init__(self, args: list[str]):
        self.args = args

    def show_help(self) -> None:
        print('Usage:')
        print('  python app.py <command> [options]')
        print()
        print('Commands:')
        print('  dataset:info    Show the dataset information')
        print('  dataset:description [--normalize]   Show the dataset description')
        print('  dataset:correlation    Show the dataset correlation')
        print('  knn:predict [--send] [--save]    Predict the target values using the KNN model')
        print()
        print('Options:')
        print('  --normalize    Normalize the dataset')
        print('  --send    Send the prediction result to the API')
        print('  --save    Save the prediction result to a file')

    def run(self):
        DATASET_FILE = 'abalone_dataset.csv'
        PREDICTION_FILE = 'abalone_app.csv'
        RESULT_KNN = 'result_knn.csv'

        try:
            if self.args[1] == 'dataset:info':
                Analysis().show_info(df=read_csv(DATASET_FILE))
            elif self.args[1] == 'dataset:description':
                Analysis().show_description(df=read_csv(DATASET_FILE), normalize='--normalize' in self.args)
            elif self.args[1] == 'dataset:correlation':
                Analysis().show_correlation(df=read_csv(DATASET_FILE))
            elif self.args[1] == 'knn:predict':
                # Train the model
                X, y = Knn().preprocess(df=read_csv(DATASET_FILE))
                model = Knn().train(X, y)

                # Predict the target values
                target = read_csv(PREDICTION_FILE)
                target = Knn().convert_sex_to_int(target)
                prediction = Knn().predict(model, df=target)

                prediction.info()
                print(prediction.describe())
                print(prediction.values)

                if '--save' in self.args:
                    target['type'] = prediction
                    target.to_csv(RESULT_KNN, index=False)

                if '--send' in self.args:
                    print(send_predictions(prediction))
            else:
                self.show_help()
        except IndexError:
            self.show_help()


if __name__ == '__main__':
    Menu(sys.argv).run()