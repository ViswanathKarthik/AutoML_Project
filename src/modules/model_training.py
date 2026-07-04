"""Model training and evaluation module"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from typing import Dict, Tuple, List
from .utils import print_subsection, print_section
import config
import shap
import pandas as pd
import numpy as np


class ModelTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self, problem_type: str = 'classification'):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        
    def _get_models(self) -> Dict:
        """Get appropriate models based on problem type"""
        if self.problem_type == 'classification':
            return {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=config.RANDOM_STATE),
                'SVM': SVC(kernel='rbf', probability=True, random_state=config.RANDOM_STATE)
            }
        else:
            return {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=config.RANDOM_STATE)
            }
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets"""
        print_subsection('Splitting Data')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y if self.problem_type == 'classification' else None
        )
        
        print(f'✓ Training set: {X_train.shape[0]} samples')
        print(f'✓ Test set: {X_test.shape[0]} samples')
        print(f'✓ Features: {X_train.shape[1]}')
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
        """Train all models"""
        print_subsection('Training Models')
        self.X_train = X_train
        models = self._get_models()
        self.models = {}
        
        for model_name, model in models.items():
            try:
                print(f'  Training {model_name}...', end=' ')
                model.fit(X_train, y_train)
                self.models[model_name] = model
                print('✓')
            except Exception as e:
                print(f'✗ Error: {str(e)}')
        
        print(f'\n✓ Trained {len(self.models)} models')
        return self.models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """Evaluate all trained models"""
        print_subsection('Evaluating Models')
        
        self.results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            if self.problem_type == 'classification':
                metrics = self._get_classification_metrics(y_test, y_pred, model_name, model)
            else:
                metrics = self._get_regression_metrics(y_test, y_pred)
            
            self.results[model_name] = metrics
            
            # Display results
            print(f'\n{model_name}:')
            for metric_name, metric_value in metrics.items():
                print(f'  {metric_name}: {metric_value:.4f}')
        
        return self.results
    
    def _get_classification_metrics(self, y_test: pd.Series, y_pred: np.ndarray, model_name: str, model) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # Add ROC-AUC if binary classification
        if len(np.unique(y_test)) == 2:
            try:
                y_pred_proba = model.predict_proba(y_test.values.reshape(-1, 1))[:, 1] if hasattr(model, 'predict_proba') else y_pred
                metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba if hasattr(model, 'predict_proba') else y_pred)
            except:
                metrics['ROC-AUC'] = 0.0
        
        return metrics
    
    def _get_regression_metrics(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'R² Score': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    def get_best_model(self) -> Tuple[str, object, Dict]:
        """Get the best performing model"""
        print_subsection('Best Model Selection')
        
        if self.problem_type == 'classification':
            # Use F1-Score as primary metric
            best_model_name = max(
                self.results.keys(),
                key=lambda x: self.results[x].get('F1-Score', 0)
            )
        else:
            # Use R² Score as primary metric
            best_model_name = max(
                self.results.keys(),
                key=lambda x: self.results[x].get('R² Score', -float('inf'))
            )
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        self.best_score = self.results[best_model_name]
        self.importance_df = None
        
        print(f'✓ Best Model: {best_model_name}')
        print(f'Metrics:')
        for metric, value in self.best_score.items():
            print(f'  {metric}: {value:.4f}')
        # SHAP Explanation
        print("\nExplaining model using SHAP")
        print("---------------------------")

        best_model = self.models[best_model_name]

        try:

            # Tree models are faster with TreeExplainer
            if best_model_name in ["Random Forest", "Gradient Boosting"]:
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(self.X_train)

                importance = np.abs(shap_values).mean(axis=0)

            else:
                explainer = shap.Explainer(best_model, self.X_train)
                shap_values = explainer(self.X_train)

                importance = np.abs(shap_values.values).mean(axis=0)

            importance_df = pd.DataFrame({
                "feature": self.X_train.columns,
                "importance": importance
            }).sort_values("importance", ascending=False)
            self.importance_df = importance_df

            print("\n Features")

            for i, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']:30s} {row['importance']:.4f}")

        except Exception as e:

            print("SHAP explanation failed:", str(e))
        
        return best_model_name, self.best_model, self.best_score, self.importance_df
    
    def cross_validate_best_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Perform cross-validation on best model"""
        print_subsection('Cross-Validation')
        
        if self.best_model is None:
            print('✗ No best model selected. Run evaluate_models first.')
            return {}
        
        if self.problem_type == 'classification':
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        
        cv_results = cross_validate(
            self.best_model, X, y,
            cv=config.CV_FOLDS,
            scoring=scoring,
            return_train_score=True
        )
        
        print(f'Cross-validation results ({config.CV_FOLDS}-fold):')
        for key in cv_results.keys():
            if 'test' in key:
                scores = cv_results[key]
                print(f'  {key}: {scores.mean():.4f} (+/- {scores.std():.4f})')
        
        return cv_results
