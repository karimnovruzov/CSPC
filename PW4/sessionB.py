import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


wine = load_wine(as_frame=True)
X = wine.data  
cultivar = wine.target  

print(" WINE DATASET DESCRIPTION ")
print("Data shape:", X.shape)


pca_raw = PCA()
pca_raw.fit(X)

print("\n--- RAW PCA EXPLAINED VARIANCE RATIO ---")
print(pca_raw.explained_variance_ratio_[:2])


pc1_loadings = pd.Series(pca_raw.components_[0], index=X.columns)
print("\n RAW PCA COMPONENT LOADINGS (PC1) ")
print(pc1_loadings.sort_values(ascending=False))
import numpy as np
import os

scaler = StandardScaler()
Z = scaler.fit_transform(X)

pca_scaled = PCA()
pca_scaled.fit(Z)

explained_var = pca_scaled.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

print("\n SCALED PCA EXPLAINED VARIANCE ")
for i, (ev, cv) in enumerate(zip(explained_var, cum_var), 1):
    print(f"PC{i}: Individual={ev:.4f} | Cumulative={cv:.4f}")

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.6, color='g', label='Individual')
plt.step(range(1, len(explained_var) + 1), cum_var, where='mid', label='Cumulative')
plt.axhline(0.9, color='r', linestyle='--', label='90% Threshold')
plt.title('Scree Plot (Wine PCA)')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.legend(loc='best')
plt.savefig("results/wine_scree_plot.png")
plt.close()

print("\nScree plot graph created and saved successfully.")

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(Z)
labels = kmeans.labels_

print("\n K-MEANS CLUSTERING ")
print("\nCluster etikets:", labels[:20])

from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(cultivar, labels)
print("\nAdjusted Rand Index:", ari)


from sklearn.metrics import silhouette_score

sil_score = silhouette_score(Z, labels)
print("\nSilhouette Score:", sil_score)



from sklearn.decomposition import PCA

pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z)

plt.figure(figsize=(8, 6))
plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=labels, cmap="viridis", edgecolor="k", alpha=0.8)
plt.title("K-Means Clusters on PCA Wine Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("results/wine_clusters_pca.png")
plt.close()

print("\nCluster visualization graph created and saved successfully.")
