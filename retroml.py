#!/usr/bin/env python3
"""
RetroML - A 90s-Style Automated Machine Learning Pipeline
=========================================================

Description:
------------
A configuration-driven machine learning pipeline with a nostalgic 90s interface.
Automates the complete ML workflow from data loading to model deployment with
a retro aesthetic. Supports classification and regression tasks with multiple
algorithms and preprocessing options.

Key Features:
------------
- JSON configuration for easy setup
- Automated data preprocessing (missing values, scaling, encoding)
- Multiple model training with auto-selection
- Comprehensive evaluation reports
- Retro-style console interface with ASCII art
- Model serialization for deployment

Usage:
------
python retroml.py <config.json>

Example Config:
--------------
{
    "dataset": {
        "path": "data.csv",
        "target_column": "target"
    },
    "problem_type": "classification",
    "preprocessing": {
        "impute_missing": "mean",
        "scale_features": true
    },
    "model": {
        "auto_select": true,
        "options": ["logistic_regression", "random_forest"]
    },
    "output": {
        "save_dir": "results/",
        "format": "pickle"
    }
}

Dependencies:
------------
Required:
- pandas
- numpy
- scikit-learn

Optional:
- xgboost (for XGBoost models)
- rich (for enhanced console output)

Classes:
--------
1. RetroMLPipeline - Main pipeline class that handles:
   - Configuration loading
   - Data preprocessing
   - Model training
   - Evaluation
   - Model saving

2. RetroMLConfig - Configuration dataclass that validates and stores:
   - Dataset information
   - Problem type
   - Preprocessing settings
   - Model options
   - Output settings

Methods:
--------
- load_data() - Loads dataset from file
- preprocess_data() - Handles missing values, scaling, encoding
- train_models() - Trains and selects best model
- evaluate_model() - Generates evaluation reports
- save_model() - Serializes trained model
- run_pipeline() - Executes complete workflow

Version:
--------
v0.1.0

License:
--------
MIT License

Copyright (c) 2025 Agile Creative Labs Inc.

Maintainers:
------------
    retroml-opensource@agilecreativelabs.com>

See Also:
---------
- GitHub: https://github.com/Agile-Creative-Labs/retroml
- Documentation: https://agile-creative-labs.github.io/retroml
"""

import json
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from retroml_config import RetroMLConfig 
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

@dataclass
class RetroMLConfig:
    """Configuration class for RetroML pipeline"""
    dataset_path: str
    target_column: str
    problem_type: str
    preprocessing: Dict[str, Any]
    model: Dict[str, Any]
    output: Dict[str, Any]

class RetroMLPipeline:
    """
    🎮 RetroML - Your 90s-Style ML Assistant 🎮
    """
    
    def __init__(self, config_path: str):
        self.console = Console() if HAS_RICH else None
        self.config = self._load_config(config_path)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.scaler = None
        self.label_encoder = None
        
        # Create output directory
        os.makedirs(self.config.output['save_dir'], exist_ok=True)
        
    def _load_config(self, config_path: str) -> RetroMLConfig:
        """Load and validate configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return RetroMLConfig(
            dataset_path=config_data['dataset']['path'],
            target_column=config_data['dataset']['target_column'],
            problem_type=config_data['problem_type'],
            preprocessing=config_data.get('preprocessing', {}),
            model=config_data.get('model', {}),
            output=config_data.get('output', {'save_dir': 'results/', 'format': 'pickle'})
        )
    
    def _print_ascii_banner(self):
        """Display retro ASCII banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║  ██████╗ ███████╗████████╗██████╗  ██████╗ ███╗   ███╗██╗    ║
║  ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗████╗ ████║██║    ║
║  ██████╔╝█████╗     ██║   ██████╔╝██║   ██║██╔████╔██║██║    ║
║  ██╔══██╗██╔══╝     ██║   ██╔══██╗██║   ██║██║╚██╔╝██║██║    ║
║  ██║  ██║███████╗   ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║███████║
║  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝═══║
║                                                                  ║
║               from Agile Creative Labs Inc.                      ║
║                     v0.1.0 BETA                                  ║      
╚══════════════════════════════════════════════════════════════════╝
        """
        if self.console:
            self.console.print(banner, style="bold cyan")
        else:
            print(banner)
    
    def _retro_print(self, message: str, style: str = "white"):
        """Print message in retro style"""
        if self.console:
            self.console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    def _loading_animation(self, message: str, duration: float = 2.0):
        """Show retro loading animation"""
        if self.console:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task(f"[cyan]{message}...", total=None)
                time.sleep(duration)
        else:
            print(f">>> {message}...")
            for i in range(int(duration * 10)):
                print("█" if i % 10 < 5 else "░", end="", flush=True)
                time.sleep(0.1)
            print(" DONE!")
    
    def load_data(self):
        """Load dataset from file"""
        self._retro_print(f"LOADING DATA FROM: {self.config.dataset_path}", "bold green")
        self._loading_animation("Reading dataset")
        
        try:
            if self.config.dataset_path.endswith('.csv'):
                self.data = pd.read_csv(self.config.dataset_path)
            elif self.config.dataset_path.endswith('.json'):
                self.data = pd.read_json(self.config.dataset_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
                
            self._retro_print(f"DATA LOADED! Shape: {self.data.shape}", "bold green")
            self._retro_print(f"Columns: {list(self.data.columns)}", "yellow")
            
        except Exception as e:
            self._retro_print(f"ERROR LOADING DATA: {str(e)}", "bold red")
            sys.exit(1)
    
    def preprocess_data(self):
        """Preprocess the dataset"""
        self._retro_print("PREPROCESSING DATA...", "bold magenta")
        
        # Separate features and target
        if self.config.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in dataset")
        
        X = self.data.drop(self.config.target_column, axis=1)
        y = self.data[self.config.target_column]
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        if self.config.preprocessing.get('impute_missing'):
            strategy = self.config.preprocessing['impute_missing']
            imputer = SimpleImputer(strategy=strategy)
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            self._retro_print(f"Missing values imputed using '{strategy}' strategy", "yellow")
        
        # Encode target for classification
        if self.config.problem_type == 'classification':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if self.config.problem_type == 'classification' else None
        )
        
        # Scale features
        if self.config.preprocessing.get('scale_features', False):
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            self._retro_print("Features scaled using StandardScaler", "yellow")
        
        self._retro_print(f"PREPROCESSING COMPLETE!", "bold green")
        self._retro_print(f"Training set: {self.X_train.shape}", "cyan")
        self._retro_print(f"Test set: {self.X_test.shape}", "cyan")
    
    def _get_default_models(self):
        """Get default models based on problem type"""
        if self.config.problem_type == 'classification':
            models = {
                'logistic_regression': LogisticRegression(random_state=42),
                'random_forest': RandomForestClassifier(random_state=42)
            }
            if HAS_XGBOOST:
                models['xgboost'] = xgb.XGBClassifier(random_state=42)
        else:  # regression
            models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(random_state=42)
            }
            if HAS_XGBOOST:
                models['xgboost'] = xgb.XGBRegressor(random_state=42)
        
        return models
    
    def train_models(self):
        """Train and select the best model"""
        self._retro_print("TRAINING MODELS...", "bold blue")
        
        models = self._get_default_models()
        
        # Filter models based on config
        if not self.config.model.get('auto_select', True):
            available_models = self.config.model.get('options', list(models.keys()))
            models = {k: v for k, v in models.items() if k in available_models}
        
        best_score = -float('inf') if self.config.problem_type == 'classification' else float('inf')
        best_model_name = None
        
        for name, model in models.items():
            self._loading_animation(f"Training {name.replace('_', ' ').title()}")
            
            try:
                model.fit(self.X_train, self.y_train)
                
                if self.config.problem_type == 'classification':
                    score = model.score(self.X_test, self.y_test)
                    self._retro_print(f"{name}: Accuracy = {score:.4f}", "green")
                    
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                        self.best_model = model
                else:
                    y_pred = model.predict(self.X_test)
                    score = mean_squared_error(self.y_test, y_pred)
                    self._retro_print(f"{name}: MSE = {score:.4f}", "green")
                    
                    if score < best_score:
                        best_score = score
                        best_model_name = name
                        self.best_model = model
                        
            except Exception as e:
                self._retro_print(f"⚠ {name} failed: {str(e)}", "red")
        
        self._retro_print(f" BEST MODEL: {best_model_name.replace('_', ' ').title()}", "bold gold")
        self._retro_print(f"Best Score: {best_score:.4f}", "bold gold")
    
    def evaluate_model(self):
        """Evaluate the best model and generate reports"""
        self._retro_print(" GENERATING EVALUATION REPORT...", "bold cyan")
        
        y_pred = self.best_model.predict(self.X_test)
        
        report_path = os.path.join(self.config.output['save_dir'], 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("🎮 RETROML EVALUATION REPORT 🎮\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Problem Type: {self.config.problem_type}\n")
            f.write(f"Dataset: {self.config.dataset_path}\n")
            f.write(f"Best Model: {type(self.best_model).__name__}\n")
            f.write("-" * 60 + "\n")
            
            if self.config.problem_type == 'classification':
                report = classification_report(self.y_test, y_pred)
                f.write("CLASSIFICATION REPORT:\n")
                f.write(report)
                f.write("\nCONFUSION MATRIX:\n")
                f.write(str(confusion_matrix(self.y_test, y_pred)))
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                f.write(f"Mean Squared Error: {mse:.4f}\n")
                f.write(f"R² Score: {r2:.4f}\n")
        
        self._retro_print(f"Report saved to: {report_path}", "green")
    
    def save_model(self):
        """Save the trained model"""
        self._retro_print("SAVING MODEL...", "bold yellow")
        
        model_path = os.path.join(self.config.output['save_dir'], 'best_model.pkl')
        
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'config': self.config
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        self._retro_print(f"Model saved to: {model_path}", "bold green")
    
    def run_pipeline(self):
        """Run the complete ML pipeline"""
        self._print_ascii_banner()
        self._retro_print("INITIALIZING RETROML PIPELINE...", "bold cyan")
        
        try:
            self.load_data()
            self.preprocess_data()
            self.train_models()
            self.evaluate_model()
            self.save_model()
            
            self._retro_print("PIPELINE COMPLETE! RADICAL! ", "bold green")
            
        except Exception as e:
            self._retro_print(f"PIPELINE FAILED: {str(e)}", "bold red")
            sys.exit(1)

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python retroml.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    pipeline = RetroMLPipeline(config_path)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
