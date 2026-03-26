import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ─────────────────────────────────────────────────────────────────
wine = fetch_ucirepo(id=186)
X_all = wine.data.features
y = wine.data.targets.squeeze()

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

# ── Feature Subsets ───────────────────────────────────────────────────────────
feature_sets = {
    'All Features (11)': list(X_all.columns),

    'Top 4 Correlated': [               # highest |corr| with quality
        'alcohol', 'density', 'volatile_acidity', 'chlorides'
    ],

    'Top 6 Correlated': [
        'alcohol', 'density', 'volatile_acidity', 'chlorides',
        'citric_acid', 'fixed_acidity'
    ],

    'No Collinear (drop density)': [    # density ~collinear with alcohol
        'alcohol', 'volatile_acidity', 'chlorides', 'citric_acid',
        'fixed_acidity', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
        'pH', 'sulphates', 'residual_sugar'
    ],

    'Acidity & Alcohol': [              # domain: taste-driving features
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'pH', 'alcohol'
    ],

    'Sulfur & Preservation': [          # domain: preservation chemistry
        'free_sulfur_dioxide', 'total_sulfur_dioxide', 'sulphates',
        'chlorides', 'alcohol'
    ],

    'Alcohol + Volatile Acidity': [     # minimal top-2 drivers
        'alcohol', 'volatile_acidity'
    ],
}

# ── Models ────────────────────────────────────────────────────────────────────
model_templates = {
    'Linear Regression': LinearRegression(),
    'Ridge (a=1.0)':     Ridge(alpha=1.0),
    'Lasso (a=0.01)':    Lasso(alpha=0.01),
    'ElasticNet':        ElasticNet(alpha=0.01, l1_ratio=0.5),
    'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                    max_depth=4, random_state=42),
}

# ── Evaluate All Combinations ─────────────────────────────────────────────────
records = []

for feat_name, features in feature_sets.items():
    X_tr = X_train_full[features]
    X_te = X_test_full[features]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    for model_name, model in model_templates.items():
        import copy
        m = copy.deepcopy(model)

        # tree models don't need scaling
        use_scaled = model_name not in ('Random Forest', 'Gradient Boosting')
        Xtr = X_tr_sc if use_scaled else X_tr.values
        Xte = X_te_sc  if use_scaled else X_te.values

        m.fit(Xtr, y_train)
        preds = m.predict(Xte)

        # 5-fold CV R2
        cv_scores = cross_val_score(m, Xtr, y_train, cv=5, scoring='r2')

        records.append({
            'Features':   feat_name,
            'Model':      model_name,
            'R2_test':    round(r2_score(y_test, preds), 4),
            'RMSE_test':  round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            'MAE_test':   round(mean_absolute_error(y_test, preds), 4),
            'CV_R2_mean': round(cv_scores.mean(), 4),
            'CV_R2_std':  round(cv_scores.std(), 4),
        })

results = pd.DataFrame(records).sort_values('CV_R2_mean', ascending=False)

print("=" * 95)
print(f"{'Features':<36} {'Model':<22} {'R2_test':>8} {'RMSE':>7} {'MAE':>7} {'CV_R2':>8} {'CV_std':>7}")
print("=" * 95)
for _, row in results.iterrows():
    print(f"{row['Features']:<36} {row['Model']:<22} {row['R2_test']:>8.4f} "
          f"{row['RMSE_test']:>7.4f} {row['MAE_test']:>7.4f} "
          f"{row['CV_R2_mean']:>8.4f} {row['CV_R2_std']:>7.4f}")

# ── Best Model ────────────────────────────────────────────────────────────────
best = results.iloc[0]
print(f"\nBest model: [{best['Model']}] with features [{best['Features']}]")
print(f"  Test R2={best['R2_test']}  RMSE={best['RMSE_test']}  CV R2={best['CV_R2_mean']} +/- {best['CV_R2_std']}")

# ── Plot 1: CV R2 Heatmap ─────────────────────────────────────────────────────
pivot = results.pivot(index='Features', columns='Model', values='CV_R2_mean')
pivot = pivot.reindex(list(feature_sets.keys()))

plt.figure(figsize=(13, 5))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGn', linewidths=0.5,
            annot_kws={'size': 9}, vmin=0.2, vmax=0.6)
plt.title('5-Fold CV R² by Feature Set & Model', fontsize=13, fontweight='bold')
plt.ylabel('Feature Set')
plt.xlabel('')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('wine_model_heatmap.png', dpi=150)
plt.close()
print("\nSaved: wine_model_heatmap.png")

# ── Plot 2: Top 15 combinations bar chart ─────────────────────────────────────
top15 = results.head(15).copy()
top15['label'] = top15['Model'] + '\n[' + top15['Features'].str[:18] + ']'

fig, ax = plt.subplots(figsize=(13, 5))
colors = ['#2a9d8f' if i == 0 else '#76b7b2' for i in range(len(top15))]
bars = ax.bar(range(len(top15)), top15['CV_R2_mean'], color=colors, edgecolor='white')
ax.errorbar(range(len(top15)), top15['CV_R2_mean'],
            yerr=top15['CV_R2_std'], fmt='none', color='#333', capsize=4, lw=1.2)
ax.set_xticks(range(len(top15)))
ax.set_xticklabels(top15['label'], fontsize=7.5)
ax.set_ylabel('CV R²')
ax.set_title('Top 15 Model + Feature Combinations (5-Fold CV R²)', fontweight='bold')
ax.set_ylim(0, 0.7)
plt.tight_layout()
plt.savefig('wine_top15_models.png', dpi=150)
plt.close()
print("Saved: wine_top15_models.png")

# ── Plot 3: Best model predicted vs actual ────────────────────────────────────
# Re-fit best model
best_features = feature_sets[best['Features']]
best_model_name = best['Model']
best_m = copy.deepcopy(model_templates[best_model_name])
use_sc = best_model_name not in ('Random Forest', 'Gradient Boosting')

sc2 = StandardScaler()
Xtr2 = sc2.fit_transform(X_train_full[best_features]) if use_sc else X_train_full[best_features].values
Xte2 = sc2.transform(X_test_full[best_features]) if use_sc else X_test_full[best_features].values
best_m.fit(Xtr2, y_train)
best_preds = best_m.predict(Xte2)

# jitter for visibility
jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(y_test))
plt.figure(figsize=(6, 5))
plt.scatter(y_test + jitter, best_preds, alpha=0.35, s=18, color='#2a9d8f')
lo, hi = y_test.min() - 0.3, y_test.max() + 0.3
plt.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect fit')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title(f'Best Model: {best_model_name}\n[{best["Features"]}]  R²={best["R2_test"]}',
          fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('wine_best_model.png', dpi=150)
plt.close()
print("Saved: wine_best_model.png")

print("\nDone.")
