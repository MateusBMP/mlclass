import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import sys

DATASET_FILE = 'barrettII_eyes_clustering.xlsx'

def plt_scatter_matrix(df, colors):
    pd.plotting.scatter_matrix(df, alpha = 0.8, c=colors, diagonal='kde')
    plt.show()

def plt_scatter_3d(df, colors):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df['dioptre'], df['AL'], df['cluster'], c=colors)

if sys.argv[1] == 'k-means':
    df = pd.read_excel(DATASET_FILE)
    df = df.drop(['ID', 'Correto'], axis=1)
    df['dioptre'] = 0.376 * (df['K2'] - df['K1'])
    df = df.drop(['K1', 'K2'], axis=1)
    df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(df)

    df['cluster'] = kmeans.labels_
    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange', 4: 'purple', 5: 'brown', 6: 'pink'}
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

elif sys.argv[1] == 'dataset:scatter':
    df = pd.read_excel(DATASET_FILE)
    plt_scatter_3d(df, colors=df['Correto'])

# data = pd.read_excel(DATASET_FILE)