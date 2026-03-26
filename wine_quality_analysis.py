import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("Loading Wine Quality dataset from UCI...")
wine = fetch_ucirepo(id=186)
X = wine.data.features
y = wine.data.targets.squeeze()

df = pd.concat([X, y], axis=1)
df.columns = [*X.columns, 'quality']

print(f"\nDataset shape: {df.shape}")
print(f"Target range: {y.min()} – {y.max()}")

# ── 2. EDA ────────────────────────────────────────────────────────────────────
print("\n── Descriptive Statistics ──")
print(df.describe().round(2).to_string())

print("\n── Missing Values ──")
print(df.isnull().sum())

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle("Wine Quality – Feature Distributions", fontsize=14, fontweight='bold')

for ax, col in zip(axes.flat, df.columns):
    ax.hist(df[col], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel('')
axes.flat[-1].set_visible(False)
plt.tight_layout()
plt.savefig('wine_distributions.png', dpi=150)
plt.close()
print("\nSaved: wine_distributions.png")

# Quality distribution
plt.figure(figsize=(7, 4))
quality_counts = df['quality'].value_counts().sort_index()
plt.bar(quality_counts.index, quality_counts.values, color='steelblue', edgecolor='white')
plt.title('Quality Score Distribution')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('wine_quality_dist.png', dpi=150)
plt.close()
print("Saved: wine_quality_dist.png")

# Correlation heatmap
plt.figure(figsize=(11, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, annot_kws={'size': 8})
plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('wine_correlation.png', dpi=150)
plt.close()
print("Saved: wine_correlation.png")

# Top correlations with quality
print("\n── Correlation with Quality (sorted) ──")
corr_quality = corr['quality'].drop('quality').sort_values(key=abs, ascending=False)
print(corr_quality.round(3).to_string())

# ── 3. Regression ─────────────────────────────────────────────────────────────
print("\n── Regression Models ──")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name != 'Random Forest' else X_train
    Xte = X_test_sc  if name != 'Random Forest' else X_test
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    results[name] = {
        'R²':   round(r2_score(y_test, preds), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        'MAE':  round(mean_absolute_error(y_test, preds), 4),
        'preds': preds,
    }
    print(f"  {name:22s}  R²={results[name]['R²']:.4f}  RMSE={results[name]['RMSE']:.4f}  MAE={results[name]['MAE']:.4f}")

# ── 4. Feature Importance (Random Forest) ─────────────────────────────────────
rf = models['Random Forest']
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue', edgecolor='white')
plt.title('Random Forest – Feature Importances', fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('wine_feature_importance.png', dpi=150)
plt.close()
print("\nSaved: wine_feature_importance.png")

# ── 5. Predicted vs Actual ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Predicted vs Actual Quality Score', fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    ax.scatter(y_test, res['preds'], alpha=0.4, s=20, color='steelblue')
    lo, hi = y_test.min() - 0.5, y_test.max() + 0.5
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
    ax.set_title(f"{name}\nR²={res['R²']}")
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

plt.tight_layout()
plt.savefig('wine_predicted_vs_actual.png', dpi=150)
plt.close()
print("Saved: wine_predicted_vs_actual.png")

print("\nDone.")
