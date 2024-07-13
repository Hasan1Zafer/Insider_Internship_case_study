import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib


data = pd.read_csv('data/processed_data.csv')


data['processed_description'] = data['processed_description'].fillna('')
data['category'] = data['category'].fillna('unknown')


category_counts = data['category'].value_counts()
data = data[data['category'].isin(category_counts[category_counts > 1].index)]


min_samples = 5
data = data[data['category'].isin(category_counts[category_counts >= min_samples].index)]


num_classes = data['category'].nunique()
print(f"Number of classes: {num_classes}")


min_train_size = num_classes
min_data_points = 2 * num_classes
if len(data) < min_data_points:
    raise ValueError(f"Insufficient data: The dataset must have at least {min_data_points} samples to ensure proper splitting.")


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['processed_description'])
y = data['category']


test_size = 1 - (min_train_size / len(data))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

print("Training set category distribution:")
print(y_train.value_counts())
print("\nTesting set category distribution:")
print(y_test.value_counts())


models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

param_grids = {
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}


best_models = {}

n_splits = max(2, min(5, min(y_train.value_counts())))
stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for model_name in models:
    print(f"\nTraining {model_name}...")
    grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=stratified_k_fold, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    y_pred = best_models[model_name].predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))


best_model = best_models['GradientBoosting']
joblib.dump(best_model, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Model training completed and saved to 'models/' directory.")
