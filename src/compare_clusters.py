# Script para testar diferentes números de clusters (k) e comparar resultados.
# Gera tabela comparativa em output/tables/cluster_comparison.csv

#import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from fuzzy import clustering_kmeans, train_mamdani
#from data import preprocess_data
#from inference import MamdaniInference
from evaluate import evaluate_model


def compare_cluster_counts(
	k_values: list[int] = [3, 4, 5, 6, 8],
	data_path: str = "data/database_treino.csv",
	test_path: str = "data/database_teste.csv",
	output_csv: str = "output/tables/cluster_comparison.csv",
) -> pd.DataFrame:
	"""
	Testa diferentes números de clusters e compara acurácia.
	Parâmetros:
		k_values: lista de k a testar
		data_path: caminho dos dados de treino
		test_path: caminho dos dados de teste
		output_csv: caminho para salvar tabela comparativa	
	Retorna:
		DataFrame com resultados
	"""
	df_test = pd.read_csv(test_path)
	X_test = df_test.drop(columns=["classe"])
	y_test = df_test["classe"]
	results = []
	for k in k_values:
		print(f"\n{'─' * 80}")
		print(f"Testando k={k} clusters")
		print(f"{'─' * 80}")
		# Clustering
		t0 = time.time()
		clustering_result = clustering_kmeans(
			data_path=data_path,
			n_clusters=k,
			random_state=42,
		)
		time_cluster = time.time() - t0
		# Treinamento Mamdani
		t0 = time.time()
		model = train_mamdani(
			df=clustering_result["df_clustered"],
			feature_cols=clustering_result["feature_cols"],
			clusters=clustering_result["df_clustered"]["cluster"],
			sigma_scale=1.0,
			output_tables_dir=Path(f"output/tables/k{k}"),
		)
		time_train = time.time() - t0
		# Avaliação
		t0 = time.time()
		metrics = evaluate_model(
			X_test,
			y_test,
			model_path=f"output/tables/k{k}/mamdani_model.json",
			verbose=False,
		)
		time_eval = time.time() - t0
		result = {
			"k": k,
			"n_rules": k,
			"accuracy": metrics["accuracy"],
			"precision": metrics["precision_weighted"],
			"recall": metrics["recall_weighted"],
			"f1_score": metrics["f1_weighted"],
			"time_cluster_s": round(time_cluster, 4),
			"time_train_s": round(time_train, 4),
			"time_eval_s": round(time_eval, 4),
			"total_time_s": round(time_cluster + time_train + time_eval, 4),
		}
		results.append(result)
		print(f"[Acurácia]: {result['accuracy']:.4f}")
		print(f"[F1-Score]: {result['f1_score']:.4f}")
		print(f"[Tempo total]: {result['total_time_s']:.4f}s")
	df_results = pd.DataFrame(results)
	df_results_sorted = df_results.sort_values("accuracy", ascending=False).reset_index(drop=True)
	Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
	df_results_sorted.to_csv(output_csv, index=False)
	print("\n" + "-" * 80)
	print("RESUMO COMPARATIVO")
	print("-" * 80)
	print(df_results_sorted.to_string(index=False))
	best_k = df_results_sorted.iloc[0]["k"]
	best_acc = df_results_sorted.iloc[0]["accuracy"]
	print(f"\n[Melhor k]: {int(best_k)} com acurácia {best_acc:.4f}")
	return df_results_sorted

if __name__ == "__main__":
	results_df = compare_cluster_counts(
		k_values=[3, 4, 5, 6, 8],
		data_path="data/database_treino.csv",
		test_path="data/database_teste.csv",
		output_csv="output/tables/cluster_comparison.csv",
	)
