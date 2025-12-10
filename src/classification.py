"""
Classification models for Mine Detection
Focus on Random Forest with extensive hyperparameter tuning
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    learning_curve, validation_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import joblib


class RandomForestOptimizer:
    """
    Random Forest classifier with hyperparameter optimization
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.baseline_model = None
        self.optimized_model = None
        self.best_params = None
        self.cv_results = None
        
    def train_baseline(self, X_train, y_train) -> RandomForestClassifier:
        """
        Train baseline Random Forest with default parameters
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained baseline model
        """
        print("Training baseline Random Forest...")
        self.baseline_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
        self.baseline_model.fit(X_train, y_train)
        print("✓ Baseline model trained")
        return self.baseline_model
    
    def grid_search_tuning(self, 
                          X_train, 
                          y_train,
                          param_grid: Optional[Dict] = None,
                          cv: int = 5) -> GridSearchCV:
        """
        Perform Grid Search for hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid (uses default if None)
            cv: Number of cross-validation folds
            
        Returns:
            GridSearchCV object with results
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        
        print(f"Starting Grid Search with {cv}-fold CV...")
        print(f"Testing {np.prod([len(v) for v in param_grid.values()])} combinations...")
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.optimized_model = grid_search.best_estimator_
        self.cv_results = grid_search.cv_results_
        
        print(f"\n✓ Grid Search completed!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def random_search_tuning(self,
                            X_train,
                            y_train,
                            param_distributions: Optional[Dict] = None,
                            n_iter: int = 50,
                            cv: int = 5) -> RandomizedSearchCV:
        """
        Perform Randomized Search for hyperparameter tuning
        Faster alternative to Grid Search
        
        Args:
            X_train: Training features
            y_train: Training target
            param_distributions: Parameter distributions
            n_iter: Number of parameter settings sampled
            cv: Number of CV folds
            
        Returns:
            RandomizedSearchCV object with results
        """
        if param_distributions is None:
            from scipy.stats import randint, uniform
            param_distributions = {
                'n_estimators': randint(50, 500),
                'max_depth': [5, 10, 15, 20, 25, None],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
        
        print(f"Starting Random Search with {n_iter} iterations...")
        
        random_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.optimized_model = random_search.best_estimator_
        self.cv_results = random_search.cv_results_
        
        print(f"\n✓ Random Search completed!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search
    
    def evaluate_model(self, 
                      model, 
                      X_test, 
                      y_test,
                      model_name: str = "Model") -> Dict:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name for printing
            
        Returns:
            Dictionary with metrics
        """
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print(f"\n{'='*60}")
        print(f"{model_name} Performance")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"{'='*60}\n")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def compare_models(self, X_test, y_test) -> pd.DataFrame:
        """
        Compare baseline vs optimized model
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Comparison dataframe
        """
        if self.baseline_model is None or self.optimized_model is None:
            raise ValueError("Both baseline and optimized models must be trained first")
        
        results = []
        
        for name, model in [("Baseline", self.baseline_model), 
                           ("Optimized", self.optimized_model)]:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
        
        comparison_df = pd.DataFrame(results)
        
        # Calculate improvement
        improvement = (
            (comparison_df.loc[1, 'Accuracy'] - comparison_df.loc[0, 'Accuracy']) 
            / comparison_df.loc[0, 'Accuracy'] * 100
        )
        
        print("\n" + "="*60)
        print("BASELINE vs OPTIMIZED COMPARISON")
        print("="*60)
        print(comparison_df.to_string(index=False))
        print(f"\nImprovement: {improvement:+.2f}%")
        print("="*60 + "\n")
        
        return comparison_df
    
    def get_feature_importance(self, 
                              feature_names: list,
                              model_type: str = 'optimized') -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            feature_names: List of feature names
            model_type: 'baseline' or 'optimized'
            
        Returns:
            DataFrame with feature importances
        """
        model = self.optimized_model if model_type == 'optimized' else self.baseline_model
        
        if model is None:
            raise ValueError(f"{model_type.capitalize()} model not trained yet")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def plot_learning_curve(self, 
                           X, 
                           y, 
                           model_type: str = 'optimized',
                           cv: int = 5):
        """
        Plot learning curve to diagnose bias/variance
        
        Args:
            X: Features
            y: Target
            model_type: 'baseline' or 'optimized'
            cv: Number of CV folds
        """
        model = self.optimized_model if model_type == 'optimized' else self.baseline_model
        
        if model is None:
            raise ValueError(f"{model_type.capitalize()} model not trained yet")
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score', marker='o')
        plt.plot(train_sizes, val_mean, label='Validation score', marker='s')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curve - {model_type.capitalize()} Random Forest')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_model(self, filepath: str, model_type: str = 'optimized'):
        """Save trained model"""
        model = self.optimized_model if model_type == 'optimized' else self.baseline_model
        joblib.dump(model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str, model_type: str = 'optimized'):
        """Load trained model"""
        model = joblib.load(filepath)
        if model_type == 'optimized':
            self.optimized_model = model
        else:
            self.baseline_model = model
        print(f"✓ Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from data_loader import create_sample_data
    from preprocessing import MineDataPreprocessor
    
    print("Creating sample data...")
    df = create_sample_data(338)
    X = df[['V', 'H', 'S']]
    y = df['M']
    
    # Preprocess
    preprocessor = MineDataPreprocessor()
    X_processed, y = preprocessor.prepare_for_random_forest(X, y)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train and optimize
    optimizer = RandomForestOptimizer(random_state=42)
    
    # Baseline
    optimizer.train_baseline(X_train, y_train)
    optimizer.evaluate_model(optimizer.baseline_model, X_test, y_test, "Baseline RF")
    
    # Optimize (using small grid for demo)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    }
    optimizer.grid_search_tuning(X_train, y_train, param_grid, cv=3)
    optimizer.evaluate_model(optimizer.optimized_model, X_test, y_test, "Optimized RF")
    
    # Compare
    comparison = optimizer.compare_models(X_test, y_test)