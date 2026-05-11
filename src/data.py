import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
 
def prepare_features(df:pd.DataFrame,random_state:int=42):
	X = df.drop(columns=["classe"])
	y = df["classe"].values
	# Split treino/teste (80% treino, 20% teste) com estratificação opcional
	X_train, X_test, y_train, y_test = train_test_split(
		X, y,
		test_size=0.2,
		stratify=y,
		random_state=random_state
	)
	X_train_num = X_train.reset_index(drop=True)
	X_test_num = X_test.reset_index(drop=True)
	X_train_final = X_train_num
	X_test_final = X_test_num
	features_names = X_train_final.columns.tolist()
	return (
		X_train_final.to_numpy(),
		X_test_final.to_numpy(),
		y_train,
		y_test,
		features_names,
	)

def _run_minimal_exploratory_analysis(
	df: pd.DataFrame,
	output_dirs: dict[str, Path],
) -> None:
	# Verificação de outliers [IF(x < mu - 3std) OR IF(x > mu + 3std)]
	outliers_df = pd.DataFrame({
		"atributo": df.select_dtypes(include=[np.number]).columns,
		"outliers_count": [
			((df[col] < (df[col].mean() - 3 * df[col].std())) | (df[col] > (df[col].mean() + 3 * df[col].std()))).sum()
			for col in df.select_dtypes(include=[np.number]).columns
		],
	})
	# Resumo geral da base
	resumo_geral = pd.DataFrame(
		[
			{
				"total_registros": int(len(df)),
				"total_atributos": int(df.shape[1]),
				"atributos_numericos": int(df.select_dtypes(include=[np.number]).shape[1]),
				"atributos_categoricos": int(df.select_dtypes(exclude=[np.number]).shape[1]),
				"valores_faltantes": int(df.isnull().sum().sum()),
				"percentual_faltantes": float((df.isnull().sum().sum() / (len(df) * df.shape[1]) * 100).round(2)),
				"outliers_totais": int(outliers_df["outliers_count"].sum()),
				"percentual_outliers": float((outliers_df["outliers_count"].sum() / len(df) * 100).round(2)),
			}
		]
	)
	resumo_geral.to_csv(
		output_dirs["tables"] / "resumo_exploratorio.csv",
		index=False,
	)
	# Estatísticas por atributo
	stats_por_atributo = pd.DataFrame({
		"atributo": df.columns,
		"tipo": df.dtypes.values,
		"valores_unicos": [df[col].nunique() for col in df.columns],
		"valores_faltantes": df.isnull().sum().values,
		"percentual_faltantes": (df.isnull().sum() / len(df) * 100).round(2).values,
		"outliers": outliers_df["outliers_count"].values,
	})
	stats_por_atributo.to_csv(
		output_dirs["tables"] / "stats_atributos.csv",
		index=False,
	)
	stats_descritivas = df.select_dtypes(include=[np.number]).describe().T
	stats_descritivas.to_csv(
		output_dirs["tables"] / "stats_descritivas.csv",
	)
	# Distribuição da classe-alvo
	class_counts = df["classe"].value_counts().sort_index()
	class_percentages = (class_counts / len(df) * 100).round(2)
	fig, axes = plt.subplots(1, 2, figsize=(12, 4))
	# Gráfico de barras
	colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
	axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', alpha=0.7)
	axes[0].set_title("Distribuição da Classe-Alvo (Contagem)", fontsize=12, fontweight='bold')
	axes[0].set_xlabel("Classe")
	axes[0].set_ylabel("Quantidade")
	axes[0].grid(axis='y', alpha=0.3)
	for i, v in enumerate(class_counts.values):
		axes[0].text(class_counts.index[i], v + 5, str(v), ha='center', fontweight='bold')
	# Gráfico de pizza
	axes[1].pie(class_percentages.values, labels=[f"Classe {i}\n({p}%)" for i, p in zip(class_counts.index, class_percentages.values)],
				autopct='', colors=colors, startangle=90)
	axes[1].set_title("Distribuição da Classe-Alvo (Porcentagem)", fontsize=12, fontweight='bold')
	plt.tight_layout()
	plt.savefig(output_dirs["plots"] / "distribuicao_classe_alvo.png", dpi=150, bbox_inches='tight')
	plt.close()
	numeric_cols = df.columns # todas são numéricas
	numeric_cols = [col for col in numeric_cols if col != "classe"]  # Exclui classe
	# Matriz de correlação
	if len(numeric_cols) > 1:
		corr_matrix = df[numeric_cols].corr()
		fig, ax = plt.subplots(figsize=(10, 8))
		im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
		ax.set_xticks(range(len(numeric_cols)))
		ax.set_yticks(range(len(numeric_cols)))
		ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
		ax.set_yticklabels(numeric_cols)
		ax.set_title("Matriz de Correlação", fontsize=12, fontweight='bold')
		plt.colorbar(im, ax=ax)
		# Adicionar valores de correlação na matriz
		for i in range(len(numeric_cols)):
			for j in range(len(numeric_cols)):
				text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
							   ha="center", va="center", color="black", fontsize=8)
		plt.tight_layout()
		plt.savefig(output_dirs["plots"] / "matriz_correlacao.png", dpi=150, bbox_inches='tight')
		plt.close()
	# Salvar resumo de valores faltantes por atributo
	missing_data = pd.DataFrame({
		"atributo": df.columns,
		"total_faltantes": df.isnull().sum().values,
		"percentual": (df.isnull().sum() / len(df) * 100).round(2).values,
	})
	missing_data = missing_data[missing_data["total_faltantes"] > 0].sort_values("total_faltantes", ascending=False)
	if len(missing_data) > 0:
		missing_data.to_csv(
			output_dirs["tables"] / "valores_faltantes.csv",
			index=False,
		)
		
def preprocess_data(data_path: str = "data/base_sintetica_media.csv") -> None:
	# Análise exploratória
	df = pd.read_csv(data_path)
	output_dirs = {
		"tables": Path("output/tables"),
		"plots": Path("output/plots"),
	}
	output_dirs["tables"].mkdir(parents=True, exist_ok=True)
	output_dirs["plots"].mkdir(parents=True, exist_ok=True)
	_run_minimal_exploratory_analysis(df=df, output_dirs=output_dirs)

	# Imputação por média
	for col in df.select_dtypes(include=[np.number]).columns:
		df[col].fillna(df[col].mean(), inplace=True)
	# Substituição de outliers por limites
	for col in df.select_dtypes(include=['float64', 'int64']).columns:
		q1 = df[col].quantile(0.25)
		q3 = df[col].quantile(0.75)
		iqr = q3 - q1
		lower_bound = q1 - 1.5 * iqr
		upper_bound = q3 + 1.5 * iqr
		df[col] = df[col].clip(lower_bound, upper_bound) 
	# Normalização Z-score
	for col in df.select_dtypes(include=[np.number]).columns:
		df[col] = (df[col] - df[col].mean()) / df[col].std()
	# Histograma (após tratamento)
	numeric_cols = df.columns # todas são numéricas
	numeric_cols = [col for col in numeric_cols if col != "classe"]  # Exclui classe
	n_cols = min(3, len(numeric_cols))
	n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
	if n_rows == 1 and n_cols == 1:
		axes = np.array([axes])
	axes = axes.flatten()
	for idx, col in enumerate(numeric_cols):
		axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
		axes[idx].set_title(f"Histograma: {col}", fontsize=10, fontweight='bold')
		axes[idx].set_xlabel(col)
		axes[idx].set_ylabel("Frequência")
		axes[idx].grid(axis='y', alpha=0.3)
	# Remover subplots vazios
	for idx in range(len(numeric_cols), len(axes)):
		fig.delaxes(axes[idx])
	plt.tight_layout()
	plt.savefig(output_dirs["plots"] / "histogramas_numericas.png", dpi=150, bbox_inches='tight')
	plt.close()
	df.to_csv('data/database_tratada.csv', index=False)
preprocess_data()