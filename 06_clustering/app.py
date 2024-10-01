import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import sys

DATASET_FILE = 'barrettII_eyes_clustering.xlsx'

def plt_scatter_matrix(df, colors):
    pd.plotting.scatter_matrix(df, alpha = 0.8, c=colors, diagonal='kde')
    plt.show()

if sys.argv[1] == 'k-means':
    df = pd.read_excel(DATASET_FILE)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df = df.drop(['ID', 'Correto', 'ACD', 'WTW'], axis=1)
    df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    kmeans.fit(df)
    df['cluster'] = kmeans.labels_
    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow', 4: 'purple'}
    plt_scatter_matrix(df, colors=df['cluster'].apply(lambda x: colors[x]))
elif sys.argv[1] == 'dataset:info':
    df = pd.read_excel(DATASET_FILE)
    print(df.info())
elif sys.argv[1] == 'dataset:description':
    df = pd.read_excel(DATASET_FILE)
    print(df.describe())
elif sys.argv[1] == 'dataset:correlation':
    df = pd.read_excel(DATASET_FILE)
    print(df.drop('Correto', axis=1).corr())

# data = pd.read_excel(DATASET_FILE)