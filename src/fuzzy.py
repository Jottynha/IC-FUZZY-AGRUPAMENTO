import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Executa K-Means sobre a base preprocessada
def clustering_kmeans(
	data_path: str = "data/database_treino.csv",
	n_clusters: int = 4,
	random_state: int = 42,
	max_iter: int = 300,
) -> dict:
	output_dirs = {
		"tables": Path("output/tables"),
		"plots": Path("output/plots"),
	}
	output_dirs["tables"].mkdir(parents=True, exist_ok=True)
	output_dirs["plots"].mkdir(parents=True, exist_ok=True)
	print(f"\n[10] Carregando dados de: {data_path}")
	df = pd.read_csv(data_path)
	print(f"Shape: {df.shape}")
	print("\n[11] Preparando atributos para agrupamento")
	feature_cols = [col for col in df.columns if col != "classe"]
	X = df[feature_cols].to_numpy()
	print(f"Atributos selecionados ({len(feature_cols)}): {feature_cols}")
	print(f"\n[12] Aplicando K-Means com k={n_clusters} clusters")
	kmeans = KMeans(
		n_clusters=n_clusters,
		random_state=random_state,
		max_iter=max_iter,
		n_init=10,
	)
	clusters = kmeans.fit_predict(X)
	df_clustered = df.copy()
	df_clustered["cluster"] = clusters
	print("\n[13] Análise dos clusters encontrados")
	cluster_counts = df_clustered["cluster"].value_counts().sort_index()
	for cluster_id, count in cluster_counts.items():
		percentage = (count / len(df_clustered)) * 100
		print(f"Cluster {cluster_id}: {count} registros ({percentage:.2f}%)")
	print("\n[14] Centróides dos clusters (valores normalizados)")
	centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
	print(centers_df.round(4).to_string())
	print("\n[15] Salvando centróides dos clusters")
	centers_path = output_dirs["tables"] / "cluster_centers.csv"
	centers_df.to_csv(centers_path, index_label="cluster_id")
	print(f"Arquivo salvo: {centers_path}")
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
	axes[0].bar(
		cluster_counts.index,
		cluster_counts.values,
		color=colors,
		edgecolor="black",
		alpha=0.7,
	)
	axes[0].set_title("Distribuição de Registros por Cluster", fontsize=12, fontweight="bold")
	axes[0].set_xlabel("Cluster ID")
	axes[0].set_ylabel("Quantidade de Registros")
	axes[0].grid(axis="y", alpha=0.3)
	for i, value in enumerate(cluster_counts.values):
		axes[0].text(cluster_counts.index[i], value + 5, str(value), ha="center", fontweight="bold")
	axes[1].pie(
		cluster_counts.values,
		labels=[f"Cluster {i}\n({count})" for i, count in zip(cluster_counts.index, cluster_counts.values)],
		autopct="%1.1f%%",
		colors=colors,
		startangle=90,
	)
	axes[1].set_title("Proporção de Clusters", fontsize=12, fontweight="bold")
	plt.tight_layout()
	plot_path = output_dirs["plots"] / "distribuicao_clusters.png"
	plt.savefig(plot_path, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"Gráfico salvo: {plot_path}")
	print("\n[16] Salvando parâmetros de clustering")
	clustering_params = {
		"algoritmo": "K-Means",
		"n_clusters": int(n_clusters),
		"random_state": random_state,
		"max_iter": max_iter,
		"inertia": float(kmeans.inertia_),
		"cluster_centers": centers_df.round(4).to_dict(orient="list"),
		"cluster_distribution": cluster_counts.to_dict(),
	}
	params_path = output_dirs["tables"] / "clustering_params.json"
	with open(params_path, "w") as file:
		json.dump(clustering_params, file, indent=2)
	print(f"Parâmetros salvos: {params_path}")
	print("\n[AGRUPAMENTO CONCLUÍDO COM SUCESSO]\n")
	return {
		"df_clustered": df_clustered,
		"centers": kmeans.cluster_centers_,
		"inertia": kmeans.inertia_,
		"kmeans_model": kmeans,
	}

if __name__ == "__main__":
	clustering_kmeans()