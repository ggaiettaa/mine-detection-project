"""
Neural Network models for Mine Detection
Multiple architectures with training visualization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class NeuralNetworkExperiment:
    """
    Neural Network experimentation with different architectures
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 random_state: int = 42):
        """
        Initialize Neural Network experiment
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            random_state: Random seed
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.random_state = random_state
        
        # Set seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.models = {}
        self.histories = {}
        
    def create_simple_model(self, 
                           learning_rate: float = 0.001,
                           name: str = "simple") -> Sequential:
        """
        Create simple 2-layer neural network
        
        Args:
            learning_rate: Learning rate for optimizer
            name: Model name
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Dense(32, activation='relu', input_dim=self.input_dim),
            Dense(16, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ], name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_medium_model(self,
                           learning_rate: float = 0.001,
                           dropout_rate: float = 0.3,
                           name: str = "medium") -> Sequential:
        """
        Create medium 3-layer network with dropout
        
        Args:
            learning_rate: Learning rate
            dropout_rate: Dropout rate
            name: Model name
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Dense(64, activation='relu', input_dim=self.input_dim),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(16, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ], name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_deep_model(self,
                         learning_rate: float = 0.001,
                         dropout_rate: float = 0.3,
                         use_batch_norm: bool = True,
                         name: str = "deep") -> Sequential:
        """
        Create deep 4-layer network with batch normalization
        
        Args:
            learning_rate: Learning rate
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            name: Model name
            
        Returns:
            Compiled Keras model
        """
        layers_list = [Dense(128, activation='relu', input_dim=self.input_dim)]
        
        if use_batch_norm:
            layers_list.append(BatchNormalization())
        layers_list.append(Dropout(dropout_rate))
        
        layers_list.extend([
            Dense(64, activation='relu'),
        ])
        if use_batch_norm:
            layers_list.append(BatchNormalization())
        layers_list.append(Dropout(dropout_rate))
        
        layers_list.extend([
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(16, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model = Sequential(layers_list, name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self,
                   model: Sequential,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   epochs: int = 100,
                   batch_size: int = 32,
                   use_early_stopping: bool = True,
                   use_lr_reduction: bool = True,
                   verbose: int = 1) -> tf.keras.callbacks.History:
        """
        Train neural network model
        
        Args:
            model: Keras model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Maximum number of epochs
            batch_size: Batch size
            use_early_stopping: Whether to use early stopping
            use_lr_reduction: Whether to reduce LR on plateau
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        callbacks_list = []
        
        if use_early_stopping:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            callbacks_list.append(early_stop)
        
        if use_lr_reduction:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
            callbacks_list.append(reduce_lr)
        
        print(f"\nTraining {model.name}...")
        print(f"Architecture: {[layer.units for layer in model.layers if hasattr(layer, 'units')]}")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        # Store model and history
        self.models[model.name] = model
        self.histories[model.name] = history
        
        print(f"✓ Training completed for {model.name}")
        
        return history
    
    def evaluate_model(self,
                      model_name: str,
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict:
        """
        Evaluate trained model
        
        Args:
            model_name: Name of model to evaluate
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")
        
        model = self.models[model_name]
        
        # Predict
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n{'='*60}")
        print(f"{model_name.upper()} Model Performance")
        print(f"{'='*60}")
        print(f"Test Loss:     {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"{'='*60}\n")
        
        return {
            'model_name': model_name,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def plot_training_history(self, 
                              model_names: Optional[List[str]] = None,
                              figsize: Tuple[int, int] = (15, 5)):
        """
        Plot training and validation metrics
        
        Args:
            model_names: List of model names to plot (all if None)
            figsize: Figure size
        """
        if model_names is None:
            model_names = list(self.histories.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for model_name in model_names:
            if model_name not in self.histories:
                print(f"Warning: {model_name} not found in histories")
                continue
            
            history = self.histories[model_name].history
            
            # Plot accuracy
            axes[0].plot(history['accuracy'], label=f'{model_name} (train)', alpha=0.8)
            axes[0].plot(history['val_accuracy'], label=f'{model_name} (val)', 
                        linestyle='--', alpha=0.8)
            
            # Plot loss
            axes[1].plot(history['loss'], label=f'{model_name} (train)', alpha=0.8)
            axes[1].plot(history['val_loss'], label=f'{model_name} (val)', 
                        linestyle='--', alpha=0.8)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compare_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare all trained models
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Comparison dataframe
        """
        results = []
        
        for model_name in self.models.keys():
            metrics = self.evaluate_model(model_name, X_test, y_test)
            
            # Get final training metrics
            history = self.histories[model_name].history
            final_train_acc = history['accuracy'][-1]
            final_val_acc = history['val_accuracy'][-1]
            
            results.append({
                'Model': model_name,
                'Train Accuracy': final_train_acc,
                'Val Accuracy': final_val_acc,
                'Test Accuracy': metrics['test_accuracy'],
                'Test Loss': metrics['test_loss'],
                'Epochs Trained': len(history['accuracy'])
            })
        
        comparison_df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("NEURAL NETWORK MODELS COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80 + "\n")
        
        return comparison_df
    
    def plot_confusion_matrices(self, 
                               X_test: np.ndarray, 
                               y_test: np.ndarray,
                               model_names: Optional[List[str]] = None):
        """
        Plot confusion matrices for all models
        
        Args:
            X_test: Test features
            y_test: Test target
            model_names: Models to plot (all if None)
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        n_models = len(model_names)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(model_names):
            model = self.models[model_name]
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name.capitalize()} Model')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        self.models[model_name].save(filepath)
        print(f"✓ Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load trained model"""
        self.models[model_name] = keras.models.load_model(filepath)
        print(f"✓ Model {model_name} loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from data_loader import create_sample_data
    from preprocessing import MineDataPreprocessor
    
    print("Creating sample data...")
    df = create_sample_data(338)
    X = df[['V', 'H', 'S']]
    y = df['M']
    
    # Preprocess for neural network
    preprocessor = MineDataPreprocessor()
    X_scaled, y_encoded = preprocessor.prepare_for_neural_network(X, y)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = \
        preprocessor.create_train_val_test_split(X_scaled, y_encoded)
    
    # Create experiment
    nn_exp = NeuralNetworkExperiment(
        input_dim=X_train.shape[1],
        num_classes=5,
        random_state=42
    )
    
    # Train simple model
    simple_model = nn_exp.create_simple_model()
    nn_exp.train_model(simple_model, X_train, y_train, X_val, y_val, 
                      epochs=50, verbose=0)
    
    # Evaluate
    nn_exp.evaluate_model('simple', X_test, y_test)
    
    # Plot history
    fig = nn_exp.plot_training_history()
    plt.show()