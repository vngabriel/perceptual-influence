import itertools
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats


def load_data(csv_files):
    data = {}
    for f in csv_files:
        df = pd.read_csv(f)
        model_name = (
            os.path.basename(f).replace(".csv", "").replace("image_metrics_test_", "")
        )
        data[model_name] = df["SSIM output"].dropna().values

    return data


def cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)

    return (u1 - u2) / s


def perform_wilcoxon_tests(data):
    results = []

    for (model1, data1), (model2, data2) in itertools.combinations(data.items(), 2):
        print(data1.shape, data2.shape)
        stat, p_value = stats.wilcoxon(data1, data2, alternative="two-sided")

        effect_size = abs(cohend(data1, data2))

        if p_value < 0.05:
            winner = model1 if np.mean(data1) > np.mean(data2) else model2
        else:
            winner = "Nenhum (sem diferença significativa)"

        ssim_1 = np.mean(data1)
        ssim_2 = np.mean(data2)

        results.append(
            (model1, model2, stat, p_value, ssim_1, ssim_2, effect_size, winner)
        )

    return results


def find_best_model(data):
    best_model = max(data, key=lambda k: data[k].mean())
    best_mean = data[best_model].mean()

    return best_model, best_mean


def plot_ssim_boxplot(data, experiments):
    box_data = [v for v in data.values()]
    model_names = list(data.keys())
    experiment_names = [experiments[k] for k in model_names]

    model_means = {model: np.mean(ssim) for model, ssim in data.items()}
    best_model_mean = max(model_means, key=model_means.get)
    best_model_mean_value = model_means[best_model_mean]

    model_medians = {model: np.median(ssim) for model, ssim in data.items()}
    best_model_median = max(model_medians, key=model_medians.get)
    best_model_median_value = model_medians[best_model_median]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=box_data, notch=True)
    plt.xticks(range(len(experiment_names)), experiment_names, rotation=0)
    plt.title("Distribution of SSIM by experiment")
    plt.xlabel("Experiments")
    plt.ylabel("SSIM")

    plt.axhline(
        y=best_model_mean_value,
        color="r",
        linestyle="--",
        label=f"{best_model_mean} (Média: {best_model_mean_value:.4f})",
    )
    plt.axhline(
        y=best_model_median_value,
        color="b",
        linestyle="-",
        label=f"{best_model_median} (Mediana: {best_model_median_value:.4f})",
    )

    plt.legend()

    plt.tight_layout()
    plt.savefig("ssim_boxplot.eps", format="eps", dpi=300)
    plt.show()


def plot_diff_histogram(data1, data2, model1_name="Modelo 1", model2_name="Modelo 2"):
    diffs = data1 - data2

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    shapiro_stat, shapiro_p = stats.shapiro(diffs)

    plt.figure(figsize=(10, 6))
    sns.histplot(diffs, kde=True, bins=30, color="skyblue", edgecolor="black")
    plt.axvline(mean_diff, color="red", linestyle="--", label=f"Média: {mean_diff:.4f}")
    plt.title(f"Distribuição das diferenças de SSIM ({model1_name} - {model2_name})")
    plt.xlabel("Diferença de SSIM")
    plt.ylabel("Contagem")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nMédia da diferença: {mean_diff:.4f}")
    print(f"Desvio padrão da diferença: {std_diff:.4f}")
    print("Teste de Shapiro-Wilk para normalidade:")
    print(f"  Estatística W = {shapiro_stat:.4f}")
    print(f"  p-valor = {shapiro_p:.4f}")

    if shapiro_p > 0.05:
        print(
            "Conclusão: A diferença parece seguir uma distribuição normal (p > 0.05)."
        )
    else:
        print("Conclusão: A diferença NÃO segue uma distribuição normal (p <= 0.05).")


def main():
    experiments = {
        "2030301631": "1",
        "3001474124": "2",
        "255671025": "3",
        "22099157": "4",
    }

    csv_files = [
        "/home/gabriel/Research/perceptual-influence-models-metrics/image_metrics_test_2030301631.csv",
        "/home/gabriel/Research/perceptual-influence-models-metrics/image_metrics_test_3001474124.csv",
        "/home/gabriel/Research/perceptual-influence-models-metrics/image_metrics_test_255671025.csv",
        "/home/gabriel/Research/perceptual-influence-models-metrics/image_metrics_test_22099157.csv",
    ]
    data = load_data(csv_files)

    best_model, best_mean = find_best_model({k: pd.Series(v) for k, v in data.items()})
    print(f"\nMelhor modelo: {best_model} (média do SSIM: {best_mean:.4f})\n")

    results = perform_wilcoxon_tests(data)

    df_results = pd.DataFrame(
        results,
        columns=[
            "Modelo 1",
            "Modelo 2",
            "Estatística Wilcoxon",
            "p-valor",
            "SSIM 1",
            "SSIM 2",
            "Cohen's d",
            "Vencedor",
        ],
    )

    print("\nResultados do Teste de Wilcoxon:")
    print(df_results)
    print()
    plot_ssim_boxplot(data, experiments)
    print()
    plot_diff_histogram(data["2030301631"], data["3001474124"])


if __name__ == "__main__":
    main()
