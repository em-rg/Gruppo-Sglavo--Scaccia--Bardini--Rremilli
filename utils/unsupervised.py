from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from preprocessing import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score, silhouette_samples
from sklearn.cluster import KMeans

def unsupervised():
    # Get the preprocessed data from preprocessing.py
    df = preprocessing(scaler_param=MinMaxScaler())

    print(df.head())

    # Apply DBSCAN to df
    dbscan = DBSCAN()
    dbscan_labels = dbscan.fit_predict(df)

    # Assegna colori ai cluster DBSCAN (grigio per outlier, colori per cluster)
    unique_labels = set(dbscan_labels)
    palette = sns.color_palette("Set2", len(unique_labels))
    color_map = {
        label: palette[i] if label != -1 else (0.6, 0.6, 0.6)  # grigio per outlier
        for i, label in enumerate(sorted(unique_labels))
    }
    colors_dbscan = [color_map[label] for label in dbscan_labels]

    # Print DBSCAN statistics: n_clusters, n_outliers
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0) # Remove outliers cluster
    n_outliers = list(dbscan_labels).count(-1)
    print(f"DBSCAN found {n_clusters} clusters and {n_outliers} outliers.")

    # Save Plot DBSCAN cluster result
    plt.figure(figsize=(10, 6))
    # Select the column absences (-4) and G3 (-1)
    plt.scatter(df.iloc[:, -4], df.iloc[:, -1], c=colors_dbscan, s=40, edgecolor='black')
    plt.title("DBSCAN Clustering")
    plt.xlabel("Absences")
    plt.ylabel("Final Grade")
    plt.grid(True)
    plt.savefig("0_dbscan_clustering.png")

    # Extra: apply PCA before DBSCAN
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)

    # Save plot PCA components
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=colors_dbscan, s=40, edgecolor='black')
    plt.title("DBSCAN Clustering (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.savefig("1_dbscan_clustering_pca.png")

    # Apply DBSCAN to PCA-transformed data
    dbscan_pca = DBSCAN()
    dbscan_labels_pca = dbscan_pca.fit_predict(df_pca)

    # Assegna colori ai cluster DBSCAN (grigio per outlier, colori per cluster)
    unique_labels_pca = set(dbscan_labels_pca)
    palette_pca = sns.color_palette("Set2", len(unique_labels_pca))
    color_map_pca = {
        label: palette_pca[i] if label != -1 else (0.6, 0.6, 0.6)  # grigio per outlier
        for i, label in enumerate(sorted(unique_labels_pca))
    }
    colors_dbscan_pca = [color_map_pca[label] for label in dbscan_labels_pca]

    # Print DBSCAN statistics: n_clusters, n_outliers
    n_clusters_pca = len(set(dbscan_labels_pca)) - (1 if -1 in dbscan_labels_pca else 0) # Remove outliers cluster
    n_outliers_pca = list(dbscan_labels_pca).count(-1)
    print(f"DBSCAN found {n_clusters_pca} clusters and {n_outliers_pca} outliers.")

    # Plot DBSCAN cluster result
    plt.figure(figsize=(10, 6))
    # Select the column component 1 and component 2
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=colors_dbscan_pca, s=40, edgecolor='black')
    plt.title("DBSCAN Clustering (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.savefig("2_dbscan_clustering_pca.png")

    # Apply the silhouette score to the clusters
    silhouette_avg = silhouette_score(df_pca, dbscan_labels_pca)
    print(f"Silhouette Score (DBSCAN, PCA): {silhouette_avg}")
    sample_silhouette_values = silhouette_samples(df_pca, dbscan_labels_pca)

    # Apply Kmean to PCA data
    kmeans_model = KMeans(n_clusters=7)
    kmeans_labels = kmeans_model.fit_predict(df_pca)

    # Plot KMeans cluster result
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels, s=40, edgecolor='black')
    plt.title("KMeans Clustering (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.savefig("3_kmeans_clustering_pca.png")

    # Apply the silhouette score to the clusters
    silhouette_avg_kmeans = silhouette_score(df_pca, kmeans_labels)
    print(f"Silhouette Score (KMeans, PCA): {silhouette_avg_kmeans}")
    sample_silhouette_values_kmeans = silhouette_samples(df_pca, kmeans_labels)

    # Apply KMeans to a series of cluster number: [4-7] from PCA data
    silhouette_scores = []
    K = range(2, 30)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_pca)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(df_pca, labels)
        silhouette_scores.append(silhouette_avg)

    # Line plot
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=list(K), y=silhouette_scores, marker='o')
    plt.title("Silhouette Score al variare di k")
    plt.xlabel("Numero di cluster (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("4_silhouette_score.png")

    # Get the most informative initial features from PCA using pca.components_
    pca_components = pca.components_
    # Plot the confusion matrix between the PCA components and the original features
    plt.figure(figsize=(10, 6))
    sns.heatmap(pca_components, annot=True, cmap='coolwarm', xticklabels=df.columns, yticklabels=[f'PC{i+1}' for i in range(pca_components.shape[0])])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("5_confusion_matrix.png")

if __name__ == "__main__":
    unsupervised()