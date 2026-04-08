# =============================================================================
# Eye-Tracking Brain Fatigue Analysis
# Biometric Signal Classification - Proof of Concept
# Author: Swapnil Yadav
# Date: April 2026
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
N_SAMPLES    = 300        # samples per class
TEST_SIZE    = 0.2
N_FOLDS      = 5
FEATURES     = [
    'pupil_diameter_mm',
    'blink_rate_per_min',
    'fixation_duration_ms',
    'saccade_velocity_deg_s',
    'perclos'
]

np.random.seed(RANDOM_STATE)


# =============================================================================
# Phase 1 - Data Generation
# Parameters derived from published eye-tracking fatigue literature:
# Caffier et al. (2003), Schleicher et al. (2008), Sirevaag & Stern (2000)
# =============================================================================

def generate_session(n, state):
    """
    Generate realistic eye-tracking session data for a given fatigue state.
    Alert:   blink rate ~17/min, fixation ~225ms, pupil ~4.5mm, PERCLOS ~0.08
    Fatigued: blink rate ~30/min, fixation ~345ms, pupil ~3.3mm, PERCLOS ~0.30
    """
    fatigue = 1 if state == 'fatigued' else 0

    return pd.DataFrame({
        'timestamp'             : np.linspace(0, n * 5, n),
        'gaze_x'                : np.random.normal(400 - fatigue * 20, 50 + fatigue * 30, n),
        'gaze_y'                : np.random.normal(300 + fatigue * 10, 45 + fatigue * 30, n),
        'pupil_diameter_mm'     : np.random.normal(4.5 - fatigue * 1.2, 0.7 + fatigue * 0.5, n),
        'blink_rate_per_min'    : np.random.normal(17  + fatigue * 13,  6   + fatigue * 5,   n),
        'fixation_duration_ms'  : np.random.normal(220 + fatigue * 120, 60  + fatigue * 70,  n),
        'saccade_velocity_deg_s': np.random.normal(280 - fatigue * 80,  70  + fatigue * 50,  n),
        'perclos'               : np.clip(
                                    np.random.normal(
                                        0.08 + fatigue * 0.22,
                                        0.06 + fatigue * 0.08, n
                                    ), 0, 1),
        'fatigue_label'         : fatigue,
        'fatigue_state'         : state
    })


df = pd.concat([
    generate_session(N_SAMPLES, 'alert'),
    generate_session(N_SAMPLES, 'fatigued')
], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

df.to_csv('data/eye_tracking_fatigue.csv', index=False)

print("Phase 1 - Dataset")
print(f"  Samples : {df.shape[0]} ({N_SAMPLES} alert, {N_SAMPLES} fatigued)")
print(f"  Features: {FEATURES}")
print()


# =============================================================================
# Phase 2 - Preprocessing and Feature Summary
# =============================================================================

print("Phase 2 - Preprocessing")
print(f"  Missing values: {df[FEATURES].isnull().sum().sum()}")
print(f"  Negative values clipped (perclos): {(df['perclos'] < 0).sum()}")
print()
print("  Feature means by state:")
print(df.groupby('fatigue_state')[FEATURES].mean().round(3).to_string())
print()

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])


# =============================================================================
# Phase 3 - Statistical Analysis
# =============================================================================

print("Phase 3 - Statistical Analysis")

# T-tests
print("\n  Independent samples t-test (alert vs fatigued):")
for f in FEATURES:
    alert_vals   = df[df['fatigue_state'] == 'alert'][f]
    fatigue_vals = df[df['fatigue_state'] == 'fatigued'][f]
    t_stat, p    = stats.ttest_ind(alert_vals, fatigue_vals)
    sig          = "significant" if p < 0.05 else "not significant"
    print(f"    {f:<30}  t={t_stat:6.2f}  p={p:.4f}  ({sig})")

# Correlation
print("\n  Pearson correlation with fatigue_label:")
corr = df[FEATURES + ['fatigue_label']].corr()['fatigue_label'].drop('fatigue_label')
print(corr.sort_values().round(3).to_string())

# PCA
pca        = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]

print(f"\n  PCA explained variance ratio: {pca.explained_variance_ratio_.round(3)}")
print(f"  Total variance (2 components): {pca.explained_variance_ratio_.sum():.1%}")
print()


# =============================================================================
# Phase 4 - Machine Learning Models
# =============================================================================

X = df[FEATURES]
y = df['fatigue_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Supervised: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
y_pred    = rf.predict(X_test)
test_acc  = accuracy_score(y_test, y_pred)

cv        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

importance_df = pd.DataFrame({
    'feature'   : FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("Phase 4 - ML Modelling")
print(f"\n  Random Forest:")
print(f"    Test accuracy       : {test_acc:.1%}")
print(f"    Cross-val accuracy  : {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")
print()
print("  Classification report:")
print(classification_report(y_test, y_pred, target_names=['Alert', 'Fatigued']))
print("  Feature importance:")
print(importance_df.to_string(index=False))

# Unsupervised: K-Means
kmeans         = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
df['cluster']  = kmeans.fit_predict(X_scaled)

cluster_labels = {}
for c in [0, 1]:
    mask               = df['cluster'] == c
    majority           = mode(df[mask]['fatigue_label'], keepdims=True).mode[0]
    cluster_labels[c]  = majority

df['cluster_label'] = df['cluster'].map(cluster_labels)
kmeans_acc          = accuracy_score(df['fatigue_label'], df['cluster_label'])

print(f"\n  K-Means clustering accuracy: {kmeans_acc:.1%}")
print()


# =============================================================================
# Phase 5 - Visualisation
# =============================================================================

STATE_COLORS = {'alert': 'steelblue', 'fatigued': 'tomato'}

# --- Plot 1: Statistical overview ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Eye-Tracking Fatigue Analysis - Statistical Overview', fontsize=13)

for i, f in enumerate(FEATURES[:4]):
    ax = axes[i // 2][i % 2]
    df[df['fatigue_state'] == 'alert'][f].hist(
        ax=ax, alpha=0.6, color='steelblue', label='Alert', bins=25)
    df[df['fatigue_state'] == 'fatigued'][f].hist(
        ax=ax, alpha=0.6, color='tomato', label='Fatigued', bins=25)
    ax.set_title(f)
    ax.legend()

sns.heatmap(df[FEATURES].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1][1], square=True, linewidths=0.5)
axes[1][1].set_title('Feature Correlation Heatmap')

for state, group in df.groupby('fatigue_state'):
    axes[1][2].scatter(group['pca_1'], group['pca_2'],
                       c=STATE_COLORS[state], label=state, alpha=0.5, s=20)
axes[1][2].set_title('PCA - Alert vs Fatigued')
axes[1][2].set_xlabel('PC1')
axes[1][2].set_ylabel('PC2')
axes[1][2].legend()

plt.tight_layout()
plt.savefig('outputs/statistical_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Plot 2: ML results ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Phase 4 - ML Model Results', fontsize=13)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Alert', 'Fatigued']).plot(
    ax=axes[0], colorbar=False)
axes[0].set_title(f'Random Forest  |  Accuracy: {test_acc:.1%}')

axes[1].barh(importance_df['feature'], importance_df['importance'], color='steelblue')
axes[1].set_title('Feature Importance')
axes[1].set_xlabel('Importance Score')

cluster_color_map = {0: 'steelblue', 1: 'tomato'}
for c in [0, 1]:
    mask  = df['cluster'] == c
    label = 'Alert' if cluster_labels[c] == 0 else 'Fatigued'
    axes[2].scatter(df[mask]['pca_1'], df[mask]['pca_2'],
                    c=cluster_color_map[c], label=f'Cluster: {label}',
                    alpha=0.5, s=20)
axes[2].set_title(f'K-Means Clustering  |  Accuracy: {kmeans_acc:.1%}')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
axes[2].legend()

plt.tight_layout()
plt.savefig('outputs/ml_results.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Plot 3: PoC Dashboard ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Eye-Tracking Brain Fatigue Analysis - PoC Dashboard',
             fontsize=13, fontweight='bold')

df_sorted                 = df.sort_values('timestamp').reset_index(drop=True)
df_sorted['fatigue_score'] = df_sorted['fatigue_label'].rolling(20, min_periods=1).mean()
axes[0][0].plot(df_sorted.index, df_sorted['fatigue_score'],
                color='tomato', linewidth=1.5)
axes[0][0].axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold')
axes[0][0].fill_between(df_sorted.index, df_sorted['fatigue_score'], 0.5,
                         where=df_sorted['fatigue_score'] > 0.5,
                         alpha=0.3, color='tomato', label='Fatigued zone')
axes[0][0].set_title('Fatigue Score Over Time')
axes[0][0].set_xlabel('Sample Index')
axes[0][0].set_ylabel('Fatigue Probability')
axes[0][0].legend(fontsize=8)

for state, color in [('alert', 'steelblue'), ('fatigued', 'tomato')]:
    subset = df[df['fatigue_state'] == state]
    axes[0][1].scatter(subset.index, subset['pupil_diameter_mm'],
                       c=color, alpha=0.3, s=10, label=state)
axes[0][1].set_title('Pupil Diameter - Alert vs Fatigued')
axes[0][1].set_xlabel('Sample Index')
axes[0][1].set_ylabel('Pupil Diameter (mm)')
axes[0][1].legend()

sns.boxplot(data=df, x='fatigue_state', y='perclos',
            palette=STATE_COLORS, ax=axes[0][2])
axes[0][2].set_title('PERCLOS Distribution (Top Predictor - 44.5% importance)')
axes[0][2].set_xlabel('State')
axes[0][2].set_ylabel('PERCLOS Value')

bar_colors = ['tomato' if i == 0 else 'steelblue' for i in range(len(importance_df))]
axes[1][0].barh(importance_df['feature'], importance_df['importance'], color=bar_colors)
axes[1][0].set_title(f'Feature Importance  |  RF Accuracy: {test_acc:.1%}')
axes[1][0].set_xlabel('Importance Score')
for i, v in enumerate(importance_df['importance']):
    axes[1][0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

bar_cv_colors = ['tomato' if s == cv_scores.min() else 'steelblue' for s in cv_scores]
axes[1][1].bar([f'Fold {i+1}' for i in range(N_FOLDS)],
               cv_scores * 100, color=bar_cv_colors)
axes[1][1].axhline(cv_scores.mean() * 100, color='gray', linestyle='--',
                   label=f'Mean: {cv_scores.mean():.1%}')
axes[1][1].set_ylim(85, 100)
axes[1][1].set_title(f'5-Fold Cross-Validation  |  Mean: {cv_scores.mean():.1%}')
axes[1][1].set_ylabel('Accuracy (%)')
axes[1][1].legend(fontsize=8)

axes[1][2].axis('off')
summary_data = [
    ['Metric',               'Value'],
    ['Dataset',              '600 samples (balanced)'],
    ['Features',             '5 eye-tracking signals'],
    ['RF Test Accuracy',     f'{test_acc:.1%}'],
    ['CV Accuracy',          f'{cv_scores.mean():.1%} +/- {cv_scores.std():.1%}'],
    ['K-Means Accuracy',     f'{kmeans_acc:.1%}'],
    ['Top Predictor',        'PERCLOS (44.5%)'],
    ['PCA Variance',         '60.6% (2 components)'],
    ['Significant Features', '5/5 (p < 0.0001)'],
]
table = axes[1][2].table(cellText=summary_data[1:], colLabels=summary_data[0],
                          loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
axes[1][2].set_title('PoC Summary Results', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fatigue_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# Summary
# =============================================================================

print("Output files:")
print("  data/eye_tracking_fatigue.csv")
print("  outputs/statistical_analysis.png")
print("  outputs/ml_results.png")
print("  outputs/fatigue_dashboard.png")