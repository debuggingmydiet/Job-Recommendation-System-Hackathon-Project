import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 ADVANCED ML MODEL TRAINING - JOB RECOMMENDATION SYSTEM")
print("="*70)
print(f"\n⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==================== LOAD DATASETS ====================
print("📂 Step 1: Loading Datasets...")
try:
    # Try large dataset first, fallback to smaller
    try:
        jobs_df = pd.read_csv('jobs_dataset_large.csv')
        candidates_df = pd.read_csv('candidates_dataset_large.csv')
        applications_df = pd.read_csv('applications_training_dataset_large.csv')
        print("   ✅ Loaded LARGE dataset")
    except:
        jobs_df = pd.read_csv('jobs_dataset.csv')
        candidates_df = pd.read_csv('candidates_dataset.csv')
        applications_df = pd.read_csv('applications_training_dataset.csv')
        print("   ✅ Loaded STANDARD dataset")
    
    print(f"   Jobs: {len(jobs_df)}")
    print(f"   Candidates: {len(candidates_df)}")
    print(f"   Applications: {len(applications_df)}")
except Exception as e:
    print(f"   ❌ Error loading datasets: {e}")
    exit(1)

# ==================== DATA PREPROCESSING ====================
print("\n⚙️ Step 2: Data Preprocessing...")

# Feature engineering - create additional features
feature_columns = ['skill_match_score', 'experience_match_score', 'cgpa_score', 'projects_score']
X = applications_df[feature_columns].copy()

# Add engineered features for better performance
X['skill_exp_interaction'] = X['skill_match_score'] * X['experience_match_score'] / 100
X['academic_performance'] = (X['cgpa_score'] + X['projects_score']) / 2
X['overall_quality'] = (X['skill_match_score'] * 0.4 + 
                        X['experience_match_score'] * 0.3 + 
                        X['cgpa_score'] * 0.15 + 
                        X['projects_score'] * 0.15)

# Target variable
y = applications_df['outcome']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"   Features: {X_scaled.shape[1]} (including engineered features)")
print(f"   Target classes: {list(label_encoder.classes_)}")

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))
print(f"   Class weights: {class_weight_dict}")

# ==================== TRAIN-TEST SPLIT ====================
print("\n✂️ Step 3: Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# ==================== MODEL TRAINING ====================
print("\n🤖 Step 4: Training Advanced Ensemble Models...\n")

# Model 1: Optimized Random Forest
print("1️⃣ Random Forest Classifier (Optimized)...")
rf_model = RandomForestClassifier(
    n_estimators=300,          # More trees
    max_depth=20,              # Deeper trees
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    bootstrap=True
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
print(f"   Accuracy: {rf_accuracy*100:.2f}% | F1-Score: {rf_f1:.4f}")

# Model 2: Optimized Gradient Boosting
print("\n2️⃣ Gradient Boosting Classifier (Optimized)...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred, average='weighted')
print(f"   Accuracy: {gb_accuracy*100:.2f}% | F1-Score: {gb_f1:.4f}")

# Model 3: Ensemble Voting Classifier
print("\n3️⃣ Ensemble Voting Classifier (Combining Both)...")
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft',  # Use probability-based voting
    weights=[1.2, 1.0]  # Slight preference to Random Forest
)
ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
print(f"   Accuracy: {ensemble_accuracy*100:.2f}% | F1-Score: {ensemble_f1:.4f}")

# ==================== MODEL COMPARISON ====================
print("\n🏆 Step 5: Model Performance Comparison")
print("\n" + "="*70)
print(f"{'Model':<30} {'Accuracy':<20} {'F1-Score':<20}")
print("="*70)
print(f"{'Random Forest':<30} {rf_accuracy*100:>6.2f}%{'':<13} {rf_f1:>6.4f}")
print(f"{'Gradient Boosting':<30} {gb_accuracy*100:>6.2f}%{'':<13} {gb_f1:>6.4f}")
print(f"{'Ensemble (Best)':<30} {ensemble_accuracy*100:>6.2f}%{'':<13} {ensemble_f1:>6.4f}")
print("="*70)

# Select best model
models = {
    'Random Forest': (rf_model, rf_accuracy, rf_f1, rf_pred),
    'Gradient Boosting': (gb_model, gb_accuracy, gb_f1, gb_pred),
    'Ensemble': (ensemble_model, ensemble_accuracy, ensemble_f1, ensemble_pred)
}

best_model_name = max(models.items(), key=lambda x: x[1][2])[0]  # Best F1
best_model, best_acc, best_f1, best_pred = models[best_model_name]

print(f"\n🥇 Best Model: {best_model_name}")
print(f"   Accuracy: {best_acc*100:.2f}%")
print(f"   F1-Score: {best_f1:.4f}")

# ==================== DETAILED EVALUATION ====================
print("\n📊 Step 6: Detailed Evaluation\n")

print("Confusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, best_pred, 
                          target_names=label_encoder.classes_, 
                          digits=4))

# Feature importance (for non-ensemble)
if best_model_name != 'Ensemble':
    print("\n🔍 Top Feature Importances:")
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in importances.iterrows():
        bar = '█' * int(row['Importance'] * 50)
        print(f"   {row['Feature']:<30} {bar} {row['Importance']:.4f}")

# ==================== CROSS-VALIDATION ====================
print("\n🔄 Step 7: Cross-Validation (5-Fold Stratified)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_scaled, y_encoded, cv=skf, 
                           scoring='accuracy', n_jobs=-1)

print(f"   Fold Accuracies: {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"   Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
print(f"   Min: {cv_scores.min()*100:.2f}% | Max: {cv_scores.max()*100:.2f}%")

# ==================== SAVE MODEL ====================
print("\n💾 Step 8: Saving Model and Components...")

joblib.dump(best_model, 'job_recommendation_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

metadata = {
    'model_type': best_model_name,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_size': {
        'total_applications': len(applications_df),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    },
    'performance': {
        'test_accuracy': float(best_acc),
        'f1_score': float(best_f1),
        'cv_mean_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'cv_min': float(cv_scores.min()),
        'cv_max': float(cv_scores.max())
    },
    'features': list(X.columns),
    'classes': list(label_encoder.classes_),
    'model_params': str(best_model.get_params()) if hasattr(best_model, 'get_params') else 'N/A'
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("   ✅ Model: job_recommendation_model.pkl")
print("   ✅ Encoder: label_encoder.pkl")
print("   ✅ Scaler: feature_scaler.pkl")
print("   ✅ Metadata: model_metadata.json")

# ==================== EXAMPLE PREDICTIONS ====================
print("\n📝 Step 9: Sample Predictions\n")

test_cases = [
    {'skill': 90, 'exp': 85, 'cgpa': 92, 'proj': 90, 'label': 'Excellent'},
    {'skill': 70, 'exp': 65, 'cgpa': 75, 'proj': 60, 'label': 'Good'},
    {'skill': 45, 'exp': 50, 'cgpa': 68, 'proj': 40, 'label': 'Average'},
    {'skill': 25, 'exp': 30, 'cgpa': 62, 'proj': 20, 'label': 'Weak'}
]

for i, tc in enumerate(test_cases, 1):
    # Prepare sample with all features
    sample = pd.DataFrame([{
        'skill_match_score': tc['skill'],
        'experience_match_score': tc['exp'],
        'cgpa_score': tc['cgpa'],
        'projects_score': tc['proj'],
        'skill_exp_interaction': tc['skill'] * tc['exp'] / 100,
        'academic_performance': (tc['cgpa'] + tc['proj']) / 2,
        'overall_quality': tc['skill']*0.4 + tc['exp']*0.3 + tc['cgpa']*0.15 + tc['proj']*0.15
    }])
    
    sample_scaled = scaler.transform(sample)
    prediction = best_model.predict(sample_scaled)
    proba = best_model.predict_proba(sample_scaled)[0]
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    
    print(f"   Case {i} [{tc['label']}]: Skill={tc['skill']}, Exp={tc['exp']}, CGPA={tc['cgpa']}, Proj={tc['proj']}")
    print(f"      → Prediction: {predicted_class.upper()} ({max(proba)*100:.1f}% confidence)")
    proba_dict = {label_encoder.classes_[i]: f"{p*100:.1f}%" for i, p in enumerate(proba)}
    print(f"      → Probabilities: {proba_dict}\n")

# ==================== FINAL SUMMARY ====================
print("="*70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\n📊 Final Model Statistics:")
print(f"   Model Type: {best_model_name}")
print(f"   Test Accuracy: {best_acc*100:.2f}%")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Cross-Validation: {cv_scores.mean()*100:.2f}% ±{cv_scores.std()*100:.2f}%")
print(f"   Training Samples: {len(X_train):,}")
print(f"   Test Samples: {len(X_test):,}")
print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n🎯 Model is production-ready and saved successfully!")
print("="*70 + "\n")
