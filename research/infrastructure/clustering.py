from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def get_sil_scores(features_df, plot=False, max_clusters=11):
    X = features_df.values 

    sil_scores = []
    max_score = 0
    best_k = 1
    for k in range(2, max_clusters):  # silhouette requires at least 2 clusters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > max_score :
            max_score = score
            best_k = k
        sil_scores.append(silhouette_score(X, labels))
    if plot :
        plt.figure(figsize=(8, 5))
        plt.plot(range(2, max_clusters), sil_scores, marker='o', color='orange')
        plt.title("Silhouette Score for Optimal K")
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.xticks(range(1, 11))
        plt.grid(True)
        plt.show()

    return (best_k, max_score)


def get_regimes(df, regime_type_id, k):
    X = df.values   # numpy array for clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    df[regime_type_id] = labels
    return df