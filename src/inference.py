# Módulo de inferência Mamdani.
# Carrega modelo treinado (JSON) e realiza predições vetorizadas.

import json
from pathlib import Path
import numpy as np
import pandas as pd
EPS = 1e-8

class MamdaniInference:
	def __init__(self, model_path: str = "output/tables/mamdani_model.json"):
		with open(model_path, "r") as f:
			self.model = json.load(f)
		self.feature_cols = self.model["feature_cols"]
		self.centers = np.array(self.model["centers"])
		self.sigmas = np.array(self.model["sigmas"])
		self.cluster_majority = {
			int(k): int(v) for k, v in self.model["cluster_majority"].items()
		}
		self.sigma_scale = self.model.get("sigma_scale", 1.0)
		self.n_rules = len(self.model["rules"])
		print(f"[Modelo Mamdani carregado]: {self.n_rules} regras, {len(self.feature_cols)} atributos")
	def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
		if isinstance(X, pd.DataFrame):
			X = X[self.feature_cols].to_numpy()
		X = np.atleast_2d(X)
		M = X.shape[0]
		z = (X[:, None, :] - self.centers[None, :, :]) / np.where(
			self.sigmas[None, :, :] <= 0, EPS, self.sigmas[None, :, :]
		)
		# Gaussiana
		mu_feat_all = np.exp(-0.5 * (z ** 2))
		# Min nos antecedentes
		mu = np.min(mu_feat_all, axis=2)
		# Agregação por classe
		possible_labels = np.unique(list(self.cluster_majority.values()))
		scores_per_class = {}
		for cls in possible_labels:
			mask = np.array([self.cluster_majority[r] == cls for r in range(self.n_rules)])
			if np.any(mask):
				scores_per_class[cls] = np.max(mu[:, mask], axis=1)
			else:
				scores_per_class[cls] = np.zeros(M)
		y_pred = np.zeros(M, dtype=int)
		for i in range(M):
			best_class = max(
				scores_per_class.items(),
				key=lambda kv: kv[1][i]
			)[0]
			y_pred[i] = best_class
		return y_pred

	def predict_proba(self, X: np.ndarray | pd.DataFrame) -> dict:
		if isinstance(X, pd.DataFrame):
			X = X[self.feature_cols].to_numpy()
		X = np.atleast_2d(X)
		M = X.shape[0]
		z = (X[:, None, :] - self.centers[None, :, :]) / np.where(
			self.sigmas[None, :, :] <= 0, EPS, self.sigmas[None, :, :]
		)
		mu_feat_all = np.exp(-0.5 * (z ** 2))
		mu = np.min(mu_feat_all, axis=2)
		possible_labels = np.unique(list(self.cluster_majority.values()))
		proba = {}
		for cls in possible_labels:
			mask = np.array([self.cluster_majority[r] == cls for r in range(self.n_rules)])
			if np.any(mask):
				proba[int(cls)] = np.max(mu[:, mask], axis=1)
			else:
				proba[int(cls)] = np.zeros(M)
		return proba

	def get_model_info(self) -> dict:
		return {
			"n_rules": self.n_rules,
			"n_features": len(self.feature_cols),
			"features": self.feature_cols,
			"rules": self.model["rules"],
			"sigma_scale": self.sigma_scale,
		}
