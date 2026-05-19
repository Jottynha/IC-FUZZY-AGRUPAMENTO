# Módulo de avaliação do modelo Mamdani.
# Calcula métricas no conjunto de teste.

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	confusion_matrix,
	classification_report,
)
from inference import MamdaniInference

def evaluate_model(
	X_test: np.ndarray | pd.DataFrame,
	y_test: np.ndarray | pd.Series,
	model_path: str = "output/tables/mamdani_model.json",
	model: dict | None = None,
	verbose: bool = True,
) -> dict:
	"""
	Avalia modelo Mamdani no conjunto de teste.
	Parâmetros:
		X_test: features de teste
		y_test: labels de teste
		model_path: caminho para o modelo JSON
		verbose: se True, imprime relatório completo
	Retorna:
		Dict com métricas: accuracy, precision, recall, f1, cm
	"""
	if model is not None:
		if isinstance(X_test, pd.DataFrame):
			X_test = X_test.to_numpy()
		y_pred = model["predict_fn"](X_test)
	else:
		inference = MamdaniInference(model_path)
		y_pred = inference.predict(X_test)
	if isinstance(y_test, pd.Series):
		y_test = y_test.to_numpy()
	acc = accuracy_score(y_test, y_pred)
	possible_labels = np.unique(y_test)
	prec = precision_score(y_test, y_pred, labels=possible_labels, average=None, zero_division=0)
	rec = recall_score(y_test, y_pred, labels=possible_labels, average=None, zero_division=0)
	f1 = f1_score(y_test, y_pred, labels=possible_labels, average=None, zero_division=0)
	prec_weighted = precision_score(y_test, y_pred, labels=possible_labels, average="weighted", zero_division=0)
	rec_weighted = recall_score(y_test, y_pred, labels=possible_labels, average="weighted", zero_division=0)
	f1_weighted = f1_score(y_test, y_pred, labels=possible_labels, average="weighted", zero_division=0)
	cm = confusion_matrix(y_test, y_pred, labels=possible_labels)	
	metrics = {
		"accuracy": float(acc),
		"precision_weighted": float(prec_weighted),
		"recall_weighted": float(rec_weighted),
		"f1_weighted": float(f1_weighted),
		"precision_per_class": {int(l): float(p) for l, p in zip(possible_labels, prec)},
		"recall_per_class": {int(l): float(r) for l, r in zip(possible_labels, rec)},
		"f1_per_class": {int(l): float(f) for l, f in zip(possible_labels, f1)},
		"confusion_matrix": cm.tolist(),
		"n_samples": len(y_test),
		"n_features": X_test.shape[1] if hasattr(X_test, "shape") else len(X_test[0]),
	}
	if verbose:
		print("\n" + "-" * 70)
		print("[AVALIAÇÃO DO MODELO MAMDANI]")
		print("-" * 70)
		print(f"\nAcurácia: {acc:.4f}")
		print(f"Precisão (média ponderada): {prec_weighted:.4f}")
		print(f"Recall (média ponderada): {rec_weighted:.4f}")
		print(f"F1-Score (média ponderada): {f1_weighted:.4f}")
		print("\nMétricas por classe:")
		for l in possible_labels:
			idx = list(possible_labels).index(l)
			print(f"  Classe {l}: Prec={prec[idx]:.4f}, Rec={rec[idx]:.4f}, F1={f1[idx]:.4f}")
		print("\nMatriz de confusão:")
		cm_df = pd.DataFrame(cm, index=possible_labels, columns=possible_labels)
		print(cm_df)
		print("\n" + "-" * 70)
	return metrics


def save_metrics(metrics: dict, output_path: str = "output/tables/evaluation_metrics.json"):
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w") as f:
		json.dump(metrics, f, indent=2)

if __name__ == "__main__":
	df_test = pd.read_csv("data/database_teste.csv")
	X_test = df_test.drop(columns=["classe"])
	y_test = df_test["classe"]
	print(f"Dataset de teste: {X_test.shape}")
	metrics = evaluate_model(X_test, y_test, verbose=True)
	save_metrics(metrics)
