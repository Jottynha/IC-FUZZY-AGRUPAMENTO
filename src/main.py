from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from data import preprocess_data
from fuzzy import clustering_kmeans, train_mamdani
from evaluate import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Rodar experimentos de agrupamento + Mamdani")
    parser.add_argument("--preprocess", action="store_true", help="Executar pré-processamento antes dos experimentos")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 4, 5, 6, 8], help="Lista de valores K a testar")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds (random_state) a usar para cada execução")
    parser.add_argument("--data", type=str, default="data/database_treino.csv", help="Caminho para dados de treino")
    parser.add_argument("--test", type=str, default="data/database_teste.csv", help="Caminho para dados de teste")
    parser.add_argument("--out", type=str, default="output/tables/experiments", help="Diretório base de saída")
    parser.add_argument("--sigma_scale", type=float, default=1.0, help="sigma_scale para o treinamento Mamdani")
    parser.add_argument("--n_init", type=int, default=10, help="Número de inicializações do KMeans")
    parser.add_argument("--init_method", type=str, default="k-means++", help="Método de inicialização do KMeans")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerância de convergência do KMeans")
    parser.add_argument("--max_iter", type=int, default=300, help="Máximo de iterações do KMeans")
    parser.add_argument("--algorithm", type=str, default=None, help="Algoritmo do KMeans")
    parser.add_argument("--save_best_model", action="store_true", help="Salvar apenas o melhor modelo encontrado")
    return parser.parse_args()

def _serialize_model(model: dict) -> dict:
    return {k: v for k, v in model.items() if k != "predict_fn"}

def run_experiments(k_values, seeds, data_path, test_path, out_base, sigma_scale,
                    n_init=10, init_method="k-means++", tol=1e-4, max_iter=300, algorithm=None,
                    save_best_model=False):
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=["classe"])
    y_test = df_test["classe"]
    records = []
    best_accuracy = -np.inf
    best_model = None
    best_info = None
    for seed in seeds:
        for k in k_values:
            print(f"\n[EXPERIMENTO] seed={seed}  k={k}")
            t0 = time.time()
            clustering_result = clustering_kmeans(
                data_path=data_path,
                n_clusters=k,
                random_state=seed,
                n_init=n_init,
                init_method=init_method,
                tol=tol,
                max_iter=max_iter,
                algorithm=algorithm,
                save_artifacts=False,
            )
            t_cluster = time.time() - t0
            t0 = time.time()
            model = train_mamdani(
                df=clustering_result["df_clustered"],
                feature_cols=clustering_result["feature_cols"],
                clusters=clustering_result["df_clustered"]["cluster"],
                sigma_scale=sigma_scale,
                save_model=False,
            )
            t_train = time.time() - t0
            t0 = time.time()
            metrics = evaluate_model(X_test, y_test, model=model, verbose=False)
            t_eval = time.time() - t0
            accuracy = float(metrics["accuracy"])
            rec = {
                "seed": int(seed),
                "k": int(k),
                "accuracy": accuracy,
                "precision_weighted": metrics.get("precision_weighted"),
                "recall_weighted": metrics.get("recall_weighted"),
                "f1_weighted": metrics.get("f1_weighted"),
                "confusion_matrix": metrics.get("confusion_matrix"),
                "time_cluster_s": round(t_cluster, 4),
                "time_train_s": round(t_train, 4),
                "time_eval_s": round(t_eval, 4),
                "total_time_s": round(t_cluster + t_train + t_eval, 4),
            }
            records.append(rec)
            print(f"[RESULT] k={k} seed={seed} acc={accuracy:.4f} total_time={rec['total_time_s']}s")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_info = {"k": int(k), "seed": int(seed), "accuracy": accuracy}
    df_res = pd.DataFrame(records)
    out_csv = out_base / "experiments_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nResumo salvo em: {out_csv}")
    if save_best_model and best_model is not None:
        best_model_path = out_base / "best_mamdani_model.json"
        with open(best_model_path, "w") as f:
            json.dump(_serialize_model(best_model), f, indent=2, ensure_ascii=False)
        print(f"Melhor modelo salvo em: {best_model_path}")
        if best_info is not None:
            print(f"Melhor configuração: k={best_info['k']} seed={best_info['seed']} acc={best_info['accuracy']:.4f}")
    return df_res


def main():
    args = parse_args()
    if args.preprocess:
        print("Executando preprocessamento")
        preprocess_data(data_path="data/base_sintetica_media.csv", random_state=42)
    run_experiments(
        k_values=args.k,
        seeds=args.seeds,
        data_path=args.data,
        test_path=args.test,
        out_base=args.out,
        sigma_scale=args.sigma_scale,
        n_init=args.n_init,
        init_method=args.init_method,
        tol=args.tol,
        max_iter=args.max_iter,
        algorithm=args.algorithm,
        save_best_model=args.save_best_model,
    )
    print("Execução concluída.")

if __name__ == "__main__":
    main()
