"""
Iris Classification with Ensemble Models
This script implements ensemble learning models for Iris dataset classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== 1. Data Loading & Exploration ====================
print("=" * 60)
print("IRIS CLASSIFICATION WITH ENSEMBLE MODELS")
print("=" * 60)

print("\n[1] Loading and Exploring Data...")
print("-" * 60)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['target_name'] = df['target'].map({0: target_names[0], 1: target_names[1], 2: target_names[2]})

print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(target_names)}")
print(f"\nClasses: {target_names}")
print(f"\nClass distribution:\n{df['target_name'].value_counts()}")

# Data statistics
print(f"\nData Statistics:\n{df.drop('target_name', axis=1).describe()}")

# ==================== 2. Data Preprocessing ====================
print("\n[2] Data Preprocessing...")
print("-" * 60)

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Train/Test ratio: {X_train.shape[0]/X_test.shape[0]:.2f}")

# ==================== 3. Model Training ====================
print("\n[3] Training Ensemble Models...")
print("-" * 60)

# Model 1: Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Model 2: Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Model 3: Decision Tree (for Voting Classifier)
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)
dt_model.fit(X_train, y_train)

# Model 4: Voting Classifier (Ensemble of Ensemble)
print("Training Voting Classifier...")
voting_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('dt', dt_model)
    ],
    voting='soft'
)
voting_model.fit(X_train, y_train)

print("✓ All models trained successfully!")

# ==================== 4. Model Evaluation ====================
print("\n[4] Model Evaluation...")
print("-" * 60)

models = {
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'Decision Tree': dt_model,
    'Voting Classifier': voting_model
}

results = {}

for model_name, model in models.items():
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'model': model
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# ==================== 5. Detailed Classification Report ====================
print("\n[5] Detailed Classification Report (Best Model: Voting Classifier)...")
print("-" * 60)

best_model_name = 'Voting Classifier'
y_pred_best = results[best_model_name]['predictions']
print(f"\n{classification_report(y_test, y_pred_best, target_names=target_names)}")

# ==================== 6. Feature Importance ====================
print("\n[6] Feature Importance Analysis...")
print("-" * 60)

# Random Forest Feature Importance
rf_importance = rf_model.feature_importances_
print(f"\nRandom Forest Feature Importance:")
for feature, importance in zip(feature_names, rf_importance):
    print(f"  {feature}: {importance:.4f}")

# Gradient Boosting Feature Importance
gb_importance = gb_model.feature_importances_
print(f"\nGradient Boosting Feature Importance:")
for feature, importance in zip(feature_names, gb_importance):
    print(f"  {feature}: {importance:.4f}")

# ==================== 7. Visualization ====================
print("\n[7] Generating Visualizations...")
print("-" * 60)

# Create results directory if it doesn't exist
import os
os.makedirs('results', exist_ok=True)

# 7.1 Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ensemble Models Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['accuracy', 'precision', 'recall', 'f1']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    model_names = list(results.keys())
    values = [results[model][metric] for model in model_names]
    
    bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel(metric.capitalize(), fontsize=11, fontweight='bold')
    ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0.85, 1.0])
    ax.axhline(y=np.mean(values), color='red', linestyle='--', linewidth=2, label='Mean')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/model_comparison.png")
plt.close()

# 7.2 Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')

for idx, (model_name, model_data) in enumerate(results.items()):
    ax = axes[idx // 2, idx % 2]
    y_pred = model_data['predictions']
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'})
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/confusion_matrices.png")
plt.close()

# 7.3 Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')

# Random Forest
ax = axes[0]
importance_df_rf = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importance
}).sort_values('Importance', ascending=False)
bars = ax.barh(importance_df_rf['Feature'], importance_df_rf['Importance'], color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Importance', fontweight='bold')
ax.set_title('Random Forest', fontsize=12, fontweight='bold')
ax.invert_yaxis()
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)

# Gradient Boosting
ax = axes[1]
importance_df_gb = pd.DataFrame({
    'Feature': feature_names,
    'Importance': gb_importance
}).sort_values('Importance', ascending=False)
bars = ax.barh(importance_df_gb['Feature'], importance_df_gb['Importance'], color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Importance', fontweight='bold')
ax.set_title('Gradient Boosting', fontsize=12, fontweight='bold')
ax.invert_yaxis()
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/feature_importance.png")
plt.close()

# 7.4 Data Distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Iris Dataset Feature Distribution', fontsize=16, fontweight='bold')

for idx, feature in enumerate(feature_names):
    ax = axes[idx // 2, idx % 2]
    
    for target, color in zip(range(3), ['#FF6B6B', '#4ECDC4', '#45B7D1']):
        mask = df['target'] == target
        ax.hist(df[mask][feature], alpha=0.6, label=target_names[target], bins=10, color=color, edgecolor='black')
    
    ax.set_xlabel(feature, fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/data_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/data_distribution.png")
plt.close()

# ==================== 8. Summary ====================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

best_model = max(results.items(), key=lambda x: x[1]['f1'])
print(f"\n🏆 Best Model: {best_model[0]}")
print(f"   F1-Score: {best_model[1]['f1']:.4f}")
print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")

print(f"\n📊 Model Rankings (by F1-Score):")
sorted_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
for rank, (model_name, metrics) in enumerate(sorted_models, 1):
    print(f"   {rank}. {model_name}: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")

print(f"\n📁 Visualizations saved to 'results/' directory:")
print(f"   - model_comparison.png")
print(f"   - confusion_matrices.png")
print(f"   - feature_importance.png")
print(f"   - data_distribution.png")

print("\n" + "=" * 60)
print("✓ Classification Complete!")
print("=" * 60 + "\n")
