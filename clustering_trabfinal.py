# -*- coding: utf-8 -*-
"""Clustering_TrabFinal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VnIvPY_h42ZIlyWJEadvEDB2JvtjXEBI
"""

!pip install pandas_ods_reader

#Basic imports
from pandas_ods_reader import read_ods
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#sklearn imports
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE     #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans    #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn.preprocessing import LabelEncoder

#plotly imports
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Configuração do Seaborn
sns.set()

# Leitura dos dados (substitua o caminho do arquivo se necessário)
data = pd.read_csv('/content/trains-uptated.csv', sep=',')

data

print(data.head())
print(data.info())

# Codificar colunas categóricas para transformar colunas com valores string para valores numéricos
categorical_columns = ['length1', 'shape1', 'load_shape1', 'length2', 'shape2', 'load_shape2',
                       'length3', 'shape3', 'load_shape3', 'length4', 'shape4', 'load_shape4']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = data[col].fillna("missing")  # Preenche valores ausentes com 'missing'
    data[col] = le.fit_transform(data[col])  # Codifica valores categóricos
    label_encoders[col] = le  # Salvar o codificador caso seja necessário decodificar mais tarde

# Preencher valores numéricos ausentes com a média
data.fillna(data.mean(numeric_only=True), inplace=True)

# Normalização dos dados
# Normaliza as colunas numéricas
scaler = StandardScaler()
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
data_scaled = pd.DataFrame(scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)

# Reintegrar as colunas categóricas codificadas, se necessário
data_scaled[categorical_columns] = data[categorical_columns]
print(data_scaled.info())

# Aplicação de Clustering (K-Means)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_scaled)
data['Cluster'] = kmeans.predict(data_scaled)

# Redução de Dimensionalidade (PCA e T-SNE)
pca_2d = PCA(n_components=2)
PCs_2d = pca_2d.fit_transform(data_scaled)
data['PC1_2d'] = PCs_2d[:, 0]
data['PC2_2d'] = PCs_2d[:, 1]

"""### Os Trens são visualizados de forma interativa no espaço 2D usando Plotly, cada ponto representa um trem. ###"""

trace1 = go.Scatter(
    x = data[data['Cluster'] == 0]['PC1_2d'],
    y = data[data['Cluster'] == 0]['PC2_2d'],
    mode = "markers",
    name = "Cluster 0",
    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
    text = None
)

trace2 = go.Scatter(
    x = data[data['Cluster'] == 1]['PC1_2d'],
    y = data[data['Cluster'] == 1]['PC2_2d'],
    mode = "markers",
    name = "Cluster 1",
    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
    text = None
)

layout = dict(title = "Visualização dos Clusters (PCA 2D)",
              xaxis= dict(title= 'PC1', ticklen= 5, zeroline= False),
              yaxis= dict(title= 'PC2', ticklen= 5, zeroline= False)
             )

fig = dict(data = [trace1, trace2], layout = layout)
iplot(fig)

"""# Gerar axiomas baseado no arquivo split_dataset.py"""

# Código do split_dataset.py

distinct_values = data['Number_of_cars'].unique()

list_df = []
for value in distinct_values:
    filtered_df = data[data['Number_of_cars'] == value]
    list_df.append(filtered_df)

list_df

from scipy.spatial.distance import cdist

# Calcula as distâncias entre os trens dentro de cada cluster
def find_similar_trains(data_scaled, cluster_labels, threshold=1.5):
    """
    Agrupa trens similares dentro de cada cluster usando a distância euclidiana.

    Parâmetros:
    - data_scaled: DataFrame escalonado com os dados.
    - cluster_labels: Série ou lista com os rótulos dos clusters.
    - threshold: Limiar para considerar dois trens similares.

    Retorno:
    - similar_groups: Dicionário com listas de DataFrames por cluster.
    """
    similar_groups = {}
    for cluster in np.unique(cluster_labels):
        # Filtra os dados do cluster atual
        cluster_data = data_scaled[cluster_labels == cluster].values

        # Calcular distâncias euclidianas
        distances = cdist(cluster_data, cluster_data, metric='euclidean')

        # Agrupar por similaridade (distância < threshold)
        groups = []
        visited = set()
        for i in range(len(distances)):
            if i not in visited:
                group = [j for j in range(len(distances)) if distances[i, j] < threshold]
                visited.update(group)
                groups.append(cluster_data[group])

        # Adicionar ao dicionário os grupos encontrados
        similar_groups[cluster] = groups

    return similar_groups

# Aplica a função de agrupamento no dataset escalonado e clusters
similarity_groups = find_similar_trains(data_scaled, data['Cluster'], threshold=1.5)

# Exibe os resultados
for cluster, groups in similarity_groups.items():
    print(f"\nCluster {cluster}:")
    for i, group in enumerate(groups):
        print(f"  Grupo {i+1} (tamanho {len(group)}):")
        print(group)

