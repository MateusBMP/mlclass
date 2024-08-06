import os
import pandas as pd
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QScrollArea, QWidget, QLabel
from PyQt6.QtGui import QPixmap

class ExpandedList(list):
    def __init__(self, *args):
        super(ExpandedList, self).__init__(args)

    def __sub__(self, other: list):
        return self.__class__(*[item for item in self if item not in other])

FILE = 'diabetes_dataset.csv'
COLUMNS = ExpandedList('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome')
IGNORED_COLUMNS = ExpandedList('Age', 'BloodPressure')

class DefaultViewer(QWidget):
    def __init__(self, files: list[str]):
        super().__init__()
        self.showing = False
        # self.image = None
        self.scroll_area: QScrollArea|None = None
        self.scroll_widget: QLabel|None = None
        self.initUI(files)

    def initUI(self, files: list[str]):
        layout = QVBoxLayout()

        # Add images to the layout
        for file in files:
            button = QPushButton(f'Show {file}', self)
            button.clicked.connect(lambda checked, file=file: self.swipe_show(file))
            layout.addWidget(button)

        # Create a scroll area
        self.scroll_area = QScrollArea()
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)
        self.setWindowTitle('Data Analytics')
        self.show()

    def swipe_show(self, file: str):
        if self.showing:
            # self.layout().removeWidget(self.image)
            # self.image.close()
            self.scroll_area.setWidget(None)  # Remove the previous content from the scroll area

        pixmap = QPixmap(file)

        # self.image = QLabel(pixmap=pixmap)
        # self.layout().addWidget(self.image)
        self.showing = True

        self.scroll_widget = QLabel()
        self.scroll_widget.setPixmap(pixmap)
        self.scroll_area.setWidget(self.scroll_widget)

if __name__ == "__main__":
    data = pd.read_csv(FILE)
    print(data.describe())

    df = pd.DataFrame(data, columns=COLUMNS - IGNORED_COLUMNS)
    files = []

    # Correlation between each column
    has_checked = []
    print(df.corr())

    # Boxplot of each column
    for column in COLUMNS - IGNORED_COLUMNS:
        df[column].plot.box()
        plt.title(f'Boxplot of {column}')
        plt.ylabel('Values')
        plt.savefig(f'{column}.png')
        files.append(f'{column}.png')
        plt.close()

    # Scatter Matrix Plot
    plt.figure()
    plt.subplots(figsize=(12, 12))
    # Filter into two groups: Outcome = 0 and Outcome = 1
    colors = {0: 'blue', 1: 'red'}
    df['Color'] = df['Outcome'].apply(lambda x: colors[x])
    pd.plotting.scatter_matrix(df, alpha=0.7, diagonal='kde', figsize=(12, 12), c=df['Color'])
    plt.suptitle('Scatter Matrix Plot')
    plt.savefig('scatter_matrix.png')
    files.append('scatter_matrix.png')
    plt.close()

    app = QApplication([])
    viewer = DefaultViewer(files)
    app.exec()

    # Delete generated files
    for file in files:
        os.remove(file)