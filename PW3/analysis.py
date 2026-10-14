import pandas as pd
df = pd.read_csv("Heart_disease_cleveland_new.csv")


print("DataFrame Shape :", df.shape)

print("\nFirst rows : ")
print(df.head())



continuous_cols = df[["age", "chol", "trestbps", "thalach"]]
print("\nContinuous columns preview:")
print(continuous_cols.head())
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

continuous_cols['age'].hist(ax=axes[0, 0], bins=15, edgecolor='black')
axes[0, 0].set_title('Age Distribution')

continuous_cols['chol'].hist(ax=axes[0, 1], bins=15, edgecolor='black')
axes[0, 1].set_title('Cholesterol Distribution')

continuous_cols['trestbps'].hist(ax=axes[1, 0], bins=15, edgecolor='black')
axes[1, 0].set_title('Resting Blood Pressure Distribution')

continuous_cols['thalach'].hist(ax=axes[1, 1], bins=15, edgecolor='black')
axes[1, 1].set_title('Max Heart Rate Distribution')

plt.tight_layout()
plt.savefig('heart_distributions.png')
print("Graph saved as 'heart_distributions.png'.")

from scipy import stats

print("\nShapiro-Wilk  Test for Normality: ")
for col in ["age", "chol", "trestbps", "thalach"]:
    stat, p_value = stats.shapiro(continuous_cols[col])
    print(f"{col}: Statistics={stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print(f" -> {col}: Normal distribution (p > 0.05)")
    else:
        print(f" -> {col}: Not normally distributed (p <= 0.05)")





        from scipy.stats import mannwhitneyu
import numpy as np


disease_thalach = df[df['target'] == 1]['thalach']
healthy_thalach = df[df['target'] == 0]['thalach']


mean_disease = disease_thalach.mean()
se_disease = disease_thalach.sem()

mean_healthy = healthy_thalach.mean()
se_healthy = healthy_thalach.sem()

print(f"Disease (1) - Mean thalach: {mean_disease:.2f} ± {se_disease:.2f}")
print(f"Healthy (0) - Mean thalach: {mean_healthy:.2f} ± {se_healthy:.2f}")


stat, p_value = mannwhitneyu(disease_thalach, healthy_thalach)
print(f"Mann-Whitney U Test: Statistics={stat:.4f}, p-value={p_value:.10f}")



plt.figure(figsize=(8, 6))
groups = ['Disease (1)', 'Healthy (0)']
means = [mean_disease, mean_healthy]
errors = [se_disease, se_healthy]

plt.bar(groups, means, yerr=errors, capsize=5, color=['coral', 'skyblue'], edgecolor='black')
plt.ylabel('Maximum Heart Rate (thalach)')
plt.title('Thalach Comparison between Disease and Healthy Groups')
plt.savefig('thalach_comparison.png')
print("Graph saved as 'thalach_comparison.png'.")


from scipy.stats import pearsonr



corr, p_val = pearsonr(df['age'], df['thalach'])
print(f"Pearson Correlation Coefficient: {corr:.4f}, p-value: {p_val:.10f}")


plt.figure(figsize=(8, 6))
plt.scatter(df['age'], df['thalach'], alpha=0.7, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate (thalach)')
plt.title('Age vs Maximum Heart Rate')
plt.savefig('age_thalach_correlation.png')
print("Graph saved as 'age_thalach_correlation.png'.")


# Dataset 2-ni yükləyirik
df_chem = pd.read_csv("chemicals_cancer.csv")

print("\n   Dataset 2: Naive Analysis    ")
print(df_chem.head())


corr_benz, p_benz = pearsonr(df_chem['benzene'], df_chem['malignancy'])
corr_cad, p_cad = pearsonr(df_chem['cadmium'], df_chem['malignancy'])

print(f"Benzene vs Malignancy Correlation: {corr_benz:.4f} (p: {p_benz})")
print(f"Cadmium vs Malignancy Correlation: {corr_cad:.4f} (p: {p_cad})")


sub_df = df_chem[(df_chem["pollution_index"] > 40) & (df_chem["pollution_index"] < 60)]


corr_benz_filtered = sub_df["benzene"].corr(sub_df["malignancy"])
corr_cad_filtered = sub_df["cadmium"].corr(sub_df["malignancy"])

print("Filtered Benzene vs Malignancy correlation:", corr_benz_filtered)
print("Filtered Cadmium vs Malignancy correlation:", corr_cad_filtered)