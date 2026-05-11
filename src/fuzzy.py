import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
EPS = 1e-8
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
		"feature_cols": feature_cols,
	}

# pertinência gaussiana por atributo (vetorial)
def _gaussian(x: np.ndarray, center: np.ndarray, sigma: np.ndarray) -> np.ndarray:
	denom = np.where(sigma <= 0, EPS, sigma)
	z = (x - center) / denom
	return np.exp(-0.5 * (z ** 2))

# Cada cluster define uma regra do tipo: 
# SE atributos estão próximos do centróide do cluster ENTÃO classe = classe majoritária.
def train_mamdani(
	df: pd.DataFrame,
	feature_cols: list[str],
	y_col: str = "classe",
	clusters: pd.Series | None = None,
	n_rules: int | None = None,
	sigma_scale: float = 1.0,
	output_tables_dir: Path = Path("output/tables"),
) -> dict:
	print("\n[17] Treinando modelo Mamdani a partir dos clusters")
	output_tables_dir.mkdir(parents=True, exist_ok=True)
	if clusters is None:
		if n_rules is None:
			raise ValueError("Se 'clusters' não for fornecido, especifique 'n_rules'.")
		kmeans = KMeans(n_clusters=n_rules, random_state=42, n_init=10)
		X_all = df[feature_cols].to_numpy()
		labels = kmeans.fit_predict(X_all)
		df = df.copy()
		df["cluster"] = labels
		centers = kmeans.cluster_centers_
	else:
		df = df.copy()
		df["cluster"] = clusters
		centers = df.groupby("cluster")[feature_cols].mean().to_numpy()
	global_std = df[feature_cols].std().replace(0, EPS).to_numpy()
	n_rules = centers.shape[0]
	sigmas = np.zeros_like(centers)
	for i in range(n_rules):
		grp = df[df["cluster"] == i][feature_cols]
		if len(grp) > 1:
			s = grp.std().to_numpy()
			s = np.where(s <= 0, global_std, s)
		else:
			s = global_std
		sigmas[i] = s * float(sigma_scale)
	cluster_majority = df.groupby("cluster")[y_col].agg(lambda s: s.mode().iloc[0]).to_dict()
	centers_arr = np.array(centers)
	sigmas_arr = np.array(sigmas)
	rules = []
	for r in range(n_rules):
		class_counts = df[df["cluster"] == r][y_col].value_counts().sort_index()
		support = int(class_counts.sum())
		majority_class = int(cluster_majority[r])
		confidence = float(class_counts.max() / support) if support else 0.0
		rules.append(
			{
				"rule_id": int(r),
				"cluster_id": int(r),
				"support": support,
				"majority_class": majority_class,
				"confidence": round(confidence, 4),
				"center": centers_arr[r].round(4).tolist(),
				"sigma": sigmas_arr[r].round(4).tolist(),
			}
		)
	model = {
		"feature_cols": feature_cols,
		"centers": centers_arr.tolist(),
		"sigmas": sigmas_arr.tolist(),
		"rules": rules,
		"cluster_majority": {int(k): int(v) for k, v in cluster_majority.items()},
		"sigma_scale": float(sigma_scale),
	}
	model_json = output_tables_dir / "mamdani_model.json"
	with open(model_json, "w") as f:
		json.dump(model, f, indent=2, ensure_ascii=False)
	print(f"Modelo Mamdani salvo: {model_json}")
	print("\n[18] Resumo das regras Mamdani")
	for rule in rules:
		print(
			f"Regra {rule['rule_id']} -> classe {rule['majority_class']} "
			f"(suporte={rule['support']}, confiança={rule['confidence']:.4f})"
		)
	def predict_fn(X_new: np.ndarray) -> np.ndarray:
		Xn = np.atleast_2d(X_new)
		M = Xn.shape[0]
		mu = np.zeros((M, n_rules))
		for r in range(n_rules):
			mu_feat = _gaussian(Xn, centers_arr[r], sigmas_arr[r])
			# Mamdani: operador min para antecedente
			mu[:, r] = np.min(mu_feat, axis=1)
		possible_labels = np.sort(df[y_col].unique())
		y_pred = np.zeros(M, dtype=int)
		for i in range(M):
			scores = {int(c): 0.0 for c in possible_labels}
			for r in range(n_rules):
				cls = int(cluster_majority[r])
				scores[cls] = max(scores[cls], float(mu[i, r]))
			y_pred[i] = max(scores.items(), key=lambda kv: kv[1])[0]
		return y_pred
	model["predict_fn"] = predict_fn
	return model

if __name__ == "__main__":
    clustering_result = clustering_kmeans()
    train_mamdani(
        df=clustering_result["df_clustered"],
        feature_cols=clustering_result["feature_cols"],
        clusters=clustering_result["df_clustered"]["cluster"],
        sigma_scale=1.0,
    )
 