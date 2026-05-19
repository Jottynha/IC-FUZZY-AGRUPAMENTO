import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Função para estratificação da amostra:
def sample_stratified_df(
    df: pd.DataFrame,
    n_samples: int,
    target_col: str = "classe",
    random_state: int = 42,
) -> pd.DataFrame:
    if n_samples < len(df):
        df, _ = train_test_split(
            df,
            train_size=n_samples,
            stratify=df[target_col],
            random_state=random_state,
        )
    return df.reset_index(drop=True)
# Função para preparar atributos e alvo além de realizar split treino/teste:
def prepare_features(df: pd.DataFrame, random_state: int = 42):
	X = df.drop(columns=["classe"])
	y = df["classe"].values
	# Split treino/teste (80% treino, 20% teste) com estratificação
	X_train, X_test, y_train, y_test = train_test_split(
		X, y,
		test_size=0.2,
		stratify=y,
		random_state=random_state
	)
	X_train_num = X_train.reset_index(drop=True)
	X_test_num = X_test.reset_index(drop=True)
	features_names = X_train_num.columns.tolist()
	return (
		X_train_num.to_numpy(),
		X_test_num.to_numpy(),
		y_train,
		y_test,
		features_names,
	)


def _run_minimal_exploratory_analysis(df: pd.DataFrame) -> None:
	print("[INFO] Resumo exploratório mínimo")
	num_cols = df.select_dtypes(include=[np.number]).columns
	outliers_total = 0
	for col in num_cols:
		lower = df[col].mean() - 3 * df[col].std()
		upper = df[col].mean() + 3 * df[col].std()
		outliers_total += int(((df[col] < lower) | (df[col] > upper)).sum())
	print(
		f"[INFO] Registros={len(df)} | atributos={df.shape[1]} | "
		f"numéricos={len(num_cols)} | faltantes={int(df.isnull().sum().sum())} | "
		f"outliers_3sigma={outliers_total}"
	)

def preprocess_data(
	data_path: str = "data/base_sintetica_media.csv",
	random_state: int = 42,
	save_preprocessing_params: bool = False,
) -> None:
	"""
	Pipeline completo do preprocessamento:
	[1] Análise exploratória (dados originais)
	[2] Split treino/teste (com estratificação)
	[3] Imputação de valores faltantes (apenas com estatísticas do treino)
	[4] Tratamento de outliers (apenas com treino)
	[5] Normalização Z-score (apenas com treino)
	[6] Salva parâmetros de preprocessamento para aplicar ao teste
	[7] Gera histogramas pós-preprocessamento
	"""
	print("\n[1] Carregando base de dados")
	df_original = pd.read_csv(data_path)
	print(f"Base carregada: {df_original.shape[0]} registros x {df_original.shape[1]} atributos")
	output_dirs = {
		"tables": Path("output/tables"),
		"plots": Path("output/plots"),
	}
	output_dirs["tables"].mkdir(parents=True, exist_ok=True)
	output_dirs["plots"].mkdir(parents=True, exist_ok=True)
	print("\n[2] Executando análise exploratória")
	_run_minimal_exploratory_analysis(df=df_original)
	# Plots exploratórios essenciais para o relatório
	class_counts = df_original["classe"].value_counts().sort_index()
	class_percentages = (class_counts / len(df_original) * 100).round(2)
	fig, axes = plt.subplots(1, 2, figsize=(12, 4))
	colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
	axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', alpha=0.7)
	axes[0].set_title("Distribuição da Classe-Alvo (Contagem)", fontsize=12, fontweight='bold')
	axes[0].set_xlabel("Classe")
	axes[0].set_ylabel("Quantidade")
	axes[0].grid(axis='y', alpha=0.3)
	for i, v in enumerate(class_counts.values):
		axes[0].text(class_counts.index[i], v + 5, str(v), ha='center', fontweight='bold')
	axes[1].pie(
		class_percentages.values,
		labels=[f"Classe {i}\n({p}%)" for i, p in zip(class_counts.index, class_percentages.values)],
		autopct='',
		colors=colors,
		startangle=90,
	)
	axes[1].set_title("Distribuição da Classe-Alvo (Porcentagem)", fontsize=12, fontweight='bold')
	plt.tight_layout()
	plt.savefig(output_dirs["plots"] / "distribuicao_classe_alvo.png", dpi=150, bbox_inches='tight')
	plt.close()
	numeric_cols = [col for col in df_original.columns if col != "classe"]
	if len(numeric_cols) > 1:
		corr_matrix = df_original[numeric_cols].corr()
		fig, ax = plt.subplots(figsize=(10, 8))
		im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
		ax.set_xticks(range(len(numeric_cols)))
		ax.set_yticks(range(len(numeric_cols)))
		ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
		ax.set_yticklabels(numeric_cols)
		ax.set_title("Matriz de Correlação", fontsize=12, fontweight='bold')
		plt.colorbar(im, ax=ax)
		for i in range(len(numeric_cols)):
			for j in range(len(numeric_cols)):
				ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha="center", va="center", color="black", fontsize=8)
		plt.tight_layout()
		plt.savefig(output_dirs["plots"] / "matriz_correlacao.png", dpi=150, bbox_inches='tight')
		plt.close()
	print("\n[3] Separando em treino (80%) e teste (20%)")
	X = df_original.drop(columns=["classe"])
	y = df_original["classe"].values
	X_train, X_test, y_train, y_test = train_test_split(
		X, y,
		test_size=0.2,
		stratify=y,
		random_state=random_state
	)
	df_train = X_train.copy()
	df_train["classe"] = y_train
	df_test = X_test.copy()
	df_test["classe"] = y_test
	print(f"Treino: {len(df_train)} registros")
	print(f"Teste: {len(df_test)} registros")
	print("\n[4] Imputando valores faltantes")
	numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
	for col in numeric_cols:
		if col != "classe":
			mean_value = df_train[col].mean()
			df_train[col] = df_train[col].fillna(mean_value)
			df_test[col] = df_test[col].fillna(mean_value)
	print(f"Imputação realizada em {len([c for c in numeric_cols if c != 'classe'])} colunas")
	print("\n[5] Tratando outliers com método IQR")
	for col in numeric_cols:
		if col != "classe":
			q1 = df_train[col].quantile(0.25)
			q3 = df_train[col].quantile(0.75)
			iqr = q3 - q1
			lower_bound = q1 - 1.5 * iqr
			upper_bound = q3 + 1.5 * iqr
			df_train[col] = df_train[col].clip(lower_bound, upper_bound)
			df_test[col] = df_test[col].clip(lower_bound, upper_bound)
	print(f"Tratamento de outliers aplicado a {len([c for c in numeric_cols if c != 'classe'])} colunas")
	print("\n[6] Normalizando com Z-score")
	for col in numeric_cols:
		if col != "classe":
			mean_value = df_train[col].mean()
			std_value = df_train[col].std()
			df_train[col] = (df_train[col] - mean_value) / std_value
			df_test[col] = (df_test[col] - mean_value) / std_value
	print(f"Normalização Z-score aplicada a {len([c for c in numeric_cols if c != 'classe'])} colunas")
	numeric_cols_plot = [col for col in numeric_cols if col != "classe"]
	n_cols = min(3, len(numeric_cols_plot))
	n_rows = (len(numeric_cols_plot) + n_cols - 1) // n_cols
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
	if n_rows == 1 and n_cols == 1:
		axes = np.array([axes])
	axes = axes.flatten()
	for idx, col in enumerate(numeric_cols_plot):
		axes[idx].hist(df_train[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
		axes[idx].set_title(f"Histograma (Treino): {col}", fontsize=10, fontweight='bold')
		axes[idx].set_xlabel(col)
		axes[idx].set_ylabel("Frequência")
		axes[idx].grid(axis='y', alpha=0.3)
	for idx in range(len(numeric_cols_plot), len(axes)):
		fig.delaxes(axes[idx])
	plt.tight_layout()
	plt.savefig(output_dirs["plots"] / "histogramas_numericas_preprocessadas.png", dpi=150, bbox_inches='tight')
	plt.close()
	if save_preprocessing_params:
		output_dir = Path("output/tables")
		output_dir.mkdir(parents=True, exist_ok=True)
		params_path = output_dir / "preprocessing_params.json"
		with open(params_path, 'w') as f:
			json.dump({"random_state": random_state}, f, indent=2)
		print(f"Parâmetros mínimos salvos em: {params_path}")
	print("\n[7] Salvando datasets preprocessados")
	df_train.to_csv('data/database_treino.csv', index=False)
	df_test.to_csv('data/database_teste.csv', index=False)
	print(f"Treino: data/database_treino.csv")
	print(f"Teste: data/database_teste.csv")
	print("\n[8] Preprocessamento concluído")
	
if __name__ == "__main__":
	preprocess_data()