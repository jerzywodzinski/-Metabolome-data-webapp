import re
from flask import Flask, request, render_template, send_file, url_for
import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.manifold import TSNE
import pacmap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

app = Flask(__name__)

file_path = 'database1.xlsx'
sheet_name = 'Contrasts'


raw_data = pd.read_excel(file_path, sheet_name=sheet_name, header=[0, 1])
raw_data.columns = [
    re.sub(r'\s*\(.*?\)', '', ' '.join(col).strip()) for col in raw_data.columns
]
data = raw_data.iloc[1:]
data.fillna(data.mean(numeric_only=True), inplace=True)
metabolites = data.iloc[:, 0]

@app.route('/')
def index():
    columns = list(data.columns[1:])
    return render_template('index.html', columns=columns)

@app.route('/plot', methods=['POST'])
def plot():
    column = request.form['column']
    if column not in data.columns:
        return "Column does not exist!", 400

    plt.figure(figsize=(10, 6))
    plt.hist(data[column].dropna(), bins=20, color='blue', edgecolor='black')
    plt.title(f'Histogram for: {column}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')

@app.route('/tsne')
def tsne_plot():
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data.iloc[:, 1:])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='blue', edgecolors='black')
    for i in range(len(tsne_result)):
        plt.text(tsne_result[i, 0], tsne_result[i, 1], str(i + 1), fontsize=8, ha='right')
    plt.title('t-SNE Dimensionality Reduction')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return send_file(buf, mimetype='image/png')

@app.route('/pacmap')
def pacmap_plot():
    pacmap_model = pacmap.PaCMAP(n_components=2)
    pacmap_result = pacmap_model.fit_transform(data.iloc[:, 1:])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pacmap_result[:, 0], pacmap_result[:, 1], c='green', edgecolors='black')
    for i in range(len(pacmap_result)):
        plt.text(pacmap_result[i, 0], pacmap_result[i, 1], str(i + 1), fontsize=8, ha='right')
    plt.title('PaCMAP Dimensionality Reduction')
    plt.xlabel('PaCMAP Component 1')
    plt.ylabel('PaCMAP Component 2')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return send_file(buf, mimetype='image/png')

@app.route('/tsne_8groups')
def tsne_8groups_plot():
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)  # Set perplexity to 2
    tsne_result = tsne.fit_transform(data.iloc[:8, 1:])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='red', edgecolors='black')
    for i in range(8):
        plt.text(tsne_result[i, 0], tsne_result[i, 1], str(i + 1), fontsize=8, ha='right')
    plt.title('t-SNE for First 8 Groups')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    if request.method == 'POST':
        try:
            n_clusters = int(request.form['n_clusters'])
        except ValueError:
            return "Invalid number of clusters!", 400

        if n_clusters < 1:
            return "Number of clusters must be at least 1!", 400

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_result = kmeans.fit_predict(data.iloc[:, 1:])
        
        silhouette_avg = silhouette_score(data.iloc[:, 1:], cluster_result)
        
        data_with_clusters = data.copy()
        data_with_clusters['Cluster'] = cluster_result

        plt.figure(figsize=(10, 6))
        for cluster in range(n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
            plt.scatter(
                cluster_data.iloc[:, 1],
                cluster_data.iloc[:, 2],
                label=f'Cluster {cluster}'
            )
            for idx in cluster_data.index:
                plt.text(
                    cluster_data.loc[idx].iloc[1],
                    cluster_data.loc[idx].iloc[2],
                    str(idx + 1),
                    fontsize=8,
                    ha='right'
                )
        
        plt.title(f'K-Means Clustering Visualization (k={n_clusters})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()

        plt.savefig('static/clustering_plot.png', format='png')
        plt.close()
        return render_template('clustering_result.html', silhouette_avg=silhouette_avg, image_url=url_for('static', filename='clustering_plot.png'), n_clusters=n_clusters)

    return render_template('clustering.html')

@app.route('/clustering_8features', methods=['GET', 'POST'])
def clustering_8features():
    if request.method == 'POST':
        try:
            n_clusters = int(request.form['n_clusters'])
        except ValueError:
            return "Invalid number of clusters!", 400

        if n_clusters < 1:
            return "Number of clusters must be at least 1!", 400

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_result = kmeans.fit_predict(data.iloc[:8, 1:])
        
        silhouette_avg = silhouette_score(data.iloc[:8, 1:], cluster_result)
        
        data_with_clusters = data.iloc[:8].copy()
        data_with_clusters['Cluster'] = cluster_result

        plt.figure(figsize=(10, 6))
        for cluster in range(n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
            plt.scatter(
                cluster_data.iloc[:, 1],  
                cluster_data.iloc[:, 2],  
                label=f'Cluster {cluster}'
            )
            for idx in cluster_data.index:
                plt.text(
                    cluster_data.loc[idx].iloc[1], 
                    cluster_data.loc[idx].iloc[2], 
                    str(idx + 1), 
                    fontsize=8,
                    ha='right'
                )
        
        plt.title(f'K-Means Clustering Visualization for First 8 Features (k={n_clusters})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()

        plt.savefig('static/clustering_8features_plot.png', format='png')
        plt.close()

        return render_template('clustering_result_8features.html', silhouette_avg=silhouette_avg, image_url=url_for('static', filename='clustering_8features_plot.png'), n_clusters=n_clusters)

    return render_template('clustering_8features.html')

if __name__ == '__main__':
    app.run(debug=True)
