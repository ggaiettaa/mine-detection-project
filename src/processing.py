"""
Data preprocessing utilities for Mine Detection dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class MineDataPreprocessor:
    """Preprocess mine detection data for ML models"""
    
    def __init__(self):
        self.scaler = None
        self.soil_encoder = None
        self.feature_names = ['V', 'H', 'S']
        
    def encode_soil_type(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode soil type to numeric values
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with encoded soil type
        """
        X = X.copy()
        
        if X['S'].dtype == 'object':
            if self.soil_encoder is None:
                self.soil_encoder = LabelEncoder()
                X['S'] = self.soil_encoder.fit_transform(X['S'])
            else:
                X['S'] = self.soil_encoder.transform(X['S'])
                
        return X
    
    def scale_features(self, 
                      X: pd.DataFrame, 
                      method: str = 'standard') -> np.ndarray:
        """
        Scale features for neural networks
        
        Args:
            X: Feature dataframe
            method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
            
        Returns:
            Scaled feature array
        """
        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        return X_scaled
    
    def prepare_for_random_forest(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for Random Forest (minimal preprocessing needed)
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            Tuple of (X_processed, y)
        """
        X_processed = self.encode_soil_type(X)
        return X_processed, y
    
    def prepare_for_neural_network(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series,
                                    scale_method: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Neural Network (encoding + scaling)
        
        Args:
            X: Feature dataframe
            y: Target series
            scale_method: Scaling method ('standard' or 'minmax')
            
        Returns:
            Tuple of (X_scaled, y_encoded)
        """
        # Encode soil type
        X_encoded = self.encode_soil_type(X)
        
        # Scale features
        X_scaled = self.scale_features(X_encoded, method=scale_method)
        
        # Convert target to 0-indexed for neural network
        y_encoded = y.values - 1  # Classes 1-5 become 0-4
        
        return X_scaled, y_encoded
    
    def create_train_test_split(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                test_size: float = 0.2,
                                random_state: int = 42,
                                stratify: bool = True) -> Tuple:
        """
        Split data into train and test sets
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of test set
            random_state: Random seed
            stratify: Whether to stratify split by target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        stratify_param = y if stratify else None
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
    
    def create_train_val_test_split(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     test_size: float = 0.2,
                                     val_size: float = 0.15,
                                     random_state: int = 42) -> Tuple:
        """
        Split data into train, validation, and test sets
        Useful for neural network training with early stopping
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining data)
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistical summary of features
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with statistics
        """
        stats = X.describe().T
        stats['missing'] = X.isnull().sum()
        stats['missing_pct'] = (X.isnull().sum() / len(X)) * 100
        
        return stats
    
    def check_class_balance(self, y: pd.Series) -> pd.DataFrame:
        """
        Check class balance in target variable
        
        Args:
            y: Target series
            
        Returns:
            DataFrame with class distribution
        """
        counts = y.value_counts().sort_index()
        percentages = (counts / len(y)) * 100
        
        balance_df = pd.DataFrame({
            'Class': counts.index,
            'Count': counts.values,
            'Percentage': percentages.values
        })
        
        return balance_df


def add_polynomial_features(X: pd.DataFrame, 
                            degree: int = 2,
                            interaction_only: bool = False) -> pd.DataFrame:
    """
    Add polynomial features (for experimentation)
    
    Args:
        X: Feature dataframe
        degree: Polynomial degree
        interaction_only: If True, only interaction features
        
    Returns:
        DataFrame with additional features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=degree, 
                              interaction_only=interaction_only,
                              include_bias=False)
    
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    
    return pd.DataFrame(X_poly, columns=feature_names, index=X.index)


def create_binary_classification_target(y: pd.Series) -> pd.Series:
    """
    Convert multi-class problem to binary (mine vs no-mine)
    
    Args:
        y: Original target (1-5)
        
    Returns:
        Binary target (0: no mine, 1: mine)
    """
    return (y > 1).astype(int)


if __name__ == "__main__":
    # Example usage
    from data_loader import create_sample_data
    
    print("Creating sample data...")
    df = create_sample_data(338)
    X = df[['V', 'H', 'S']]
    y = df['M']
    
    print("\n" + "="*60)
    print("PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    preprocessor = MineDataPreprocessor()
    
    # For Random Forest
    print("\n1. Preparing data for Random Forest...")
    X_rf, y_rf = preprocessor.prepare_for_random_forest(X, y)
    print(f"   Shape: {X_rf.shape}")
    print(f"   Features remain as DataFrame (no scaling needed)")
    
    # For Neural Network
    print("\n2. Preparing data for Neural Network...")
    X_nn, y_nn = preprocessor.prepare_for_neural_network(X, y)
    print(f"   Shape: {X_nn.shape}")
    print(f"   Data type: {type(X_nn)}")
    print(f"   Target classes: {np.unique(y_nn)}")
    
    # Split data
    print("\n3. Creating train/val/test split...")
    X_train, X_val, X_test, y_train, y_val, y_test = \
        preprocessor.create_train_val_test_split(X_nn, y_nn)
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Val set:   {X_val.shape[0]} samples")
    print(f"   Test set:  {X_test.shape[0]} samples")
    
    # Check balance
    print("\n4. Class balance:")
    balance = preprocessor.check_class_balance(y)
    print(balance.to_string(index=False))

