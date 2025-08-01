#!/usr/bin/env python3
"""
// Copyright (c) Agile Creative Labs Inc.
// Licensed under the MIT License.

Complete RetroML Project Creator
================================
Creates the entire RetroML project structure with all files
"""

import os
import json
from pathlib import Path

def create_complete_project():
    """Create the complete RetroML project structure"""
    
    print("🎮 CREATING RETROML PROJECT - TOTALLY RADICAL! 🎮")
    print("=" * 60)
    
    # Create main retroml.py (main pipeline)
    retroml_code = '''#!/usr/bin/env python3
"""
RetroML - A 90s-Style Automated Machine Learning Pipeline
=========================================================
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
║  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝══
║                                                                ║
║             Your 90s-Style ML Assistant                        ║
║                     v0.1.0 BETA                                ║  
║             Agile Creative Labs Inc.                           ║      
╚════════════════════════════════════════════════════════════════╝
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
        self._retro_print(f"🎯 LOADING DATA FROM: {self.config.dataset_path}", "bold green")
        self._loading_animation("Reading dataset")
        
        try:
            if self.config.dataset_path.endswith('.csv'):
                self.data = pd.read_csv(self.config.dataset_path)
            elif self.config.dataset_path.endswith('.json'):
                self.data = pd.read_json(self.config.dataset_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
                
            self._retro_print(f"✅ DATA LOADED! Shape: {self.data.shape}", "bold green")
            self._retro_print(f"📊 Columns: {list(self.data.columns)}", "yellow")
            
        except Exception as e:
            self._retro_print(f"❌ ERROR LOADING DATA: {str(e)}", "bold red")
            sys.exit(1)
    
    def preprocess_data(self):
        """Preprocess the dataset"""
        self._retro_print("🔧 PREPROCESSING DATA...", "bold magenta")
        
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
            self._retro_print(f"🔄 Missing values imputed using '{strategy}' strategy", "yellow")
        
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
            self._retro_print("📏 Features scaled using StandardScaler", "yellow")
        
        self._retro_print(f"✅ PREPROCESSING COMPLETE!", "bold green")
        self._retro_print(f"🎯 Training set: {self.X_train.shape}", "cyan")
        self._retro_print(f"🎯 Test set: {self.X_test.shape}", "cyan")
    
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
        self._retro_print("🚀 TRAINING MODELS...", "bold blue")
        
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
                    self._retro_print(f"📈 {name}: Accuracy = {score:.4f}", "green")
                    
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                        self.best_model = model
                else:
                    y_pred = model.predict(self.X_test)
                    score = mean_squared_error(self.y_test, y_pred)
                    self._retro_print(f"📈 {name}: MSE = {score:.4f}", "green")
                    
                    if score < best_score:
                        best_score = score
                        best_model_name = name
                        self.best_model = model
                        
            except Exception as e:
                self._retro_print(f"⚠️  {name} failed: {str(e)}", "red")
        
        self._retro_print(f"🏆 BEST MODEL: {best_model_name.replace('_', ' ').title()}", "bold gold")
        self._retro_print(f"🎯 Best Score: {best_score:.4f}", "bold gold")
    
    def evaluate_model(self):
        """Evaluate the best model and generate reports"""
        self._retro_print("📊 GENERATING EVALUATION REPORT...", "bold cyan")
        
        y_pred = self.best_model.predict(self.X_test)
        
        report_path = os.path.join(self.config.output['save_dir'], 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\\n")
            f.write("🎮 RETROML EVALUATION REPORT 🎮\\n")
            f.write("=" * 60 + "\\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Problem Type: {self.config.problem_type}\\n")
            f.write(f"Dataset: {self.config.dataset_path}\\n")
            f.write(f"Best Model: {type(self.best_model).__name__}\\n")
            f.write("-" * 60 + "\\n")
            
            if self.config.problem_type == 'classification':
                report = classification_report(self.y_test, y_pred)
                f.write("CLASSIFICATION REPORT:\\n")
                f.write(report)
                f.write("\\nCONFUSION MATRIX:\\n")
                f.write(str(confusion_matrix(self.y_test, y_pred)))
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                f.write(f"Mean Squared Error: {mse:.4f}\\n")
                f.write(f"R² Score: {r2:.4f}\\n")
        
        self._retro_print(f"📋 Report saved to: {report_path}", "green")
    
    def save_model(self):
        """Save the trained model"""
        self._retro_print("💾 SAVING MODEL...", "bold yellow")
        
        model_path = os.path.join(self.config.output['save_dir'], 'best_model.pkl')
        
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'config': self.config
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        self._retro_print(f"✅ Model saved to: {model_path}", "bold green")
    
    def run_pipeline(self):
        """Run the complete ML pipeline"""
        self._print_ascii_banner()
        self._retro_print("🎮 INITIALIZING RETROML PIPELINE...", "bold cyan")
        
        try:
            self.load_data()
            self.preprocess_data()
            self.train_models()
            self.evaluate_model()
            self.save_model()
            
            self._retro_print("🎉 PIPELINE COMPLETE! RADICAL! 🎉", "bold green")
            
        except Exception as e:
            self._retro_print(f"💥 PIPELINE FAILED: {str(e)}", "bold red")
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
'''
    
    # Write main pipeline file
    with open('retroml.py', 'w') as f:
        f.write(retroml_code)
    print("✅ Created retroml.py")
    
    # Create requirements.txt
    requirements = """# RetroML Requirements
# Core ML Libraries
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0

# Optional ML Libraries
xgboost>=1.6.0
lightgbm>=3.3.0

# UI and Visualization
rich>=12.0.0
colorama>=0.4.0

# Configuration and Validation
pydantic>=1.10.0
jsonschema>=4.0.0

# Utilities
tqdm>=4.64.0
python-dateutil>=2.8.0

# Optional: Advanced features
# mlflow>=2.0.0
# optuna>=3.0.0
# shap>=0.41.0
# plotly>=5.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ Created requirements.txt")
    
    # Create project directories
    dirs = ["data", "results", "configs", "tests", "docs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Create sample datasets
    import pandas as pd
    import numpy as np
    
    # Customer churn dataset
    np.random.seed(42)
    n_samples = 1000
    
    churn_data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(40, 15, n_samples).astype(int),
        'tenure_months': np.random.exponential(24, n_samples).astype(int),
        'monthly_charges': np.random.normal(65, 20, n_samples).round(2),
        'total_charges': np.random.normal(1500, 800, n_samples).round(2),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
        'online_security': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tech_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    }
    
    # Create target with logic
    churn_prob = (
        (churn_data['monthly_charges'] > 80) * 0.3 +
        (churn_data['contract_type'] == 'Month-to-month') * 0.4 +
        (churn_data['tenure_months'] < 12) * 0.3 +
        np.random.normal(0, 0.1, n_samples)
    )
    churn_data['churn'] = (churn_prob > 0.5).astype(int)
    
    pd.DataFrame(churn_data).to_csv('data/customers.csv', index=False)
    print("✅ Created data/customers.csv")
    
    # House prices dataset
    house_data = {
        'house_id': range(1, n_samples + 1),
        'bedrooms': np.random.choice([2, 3, 4, 5], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        'bathrooms': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        'sqft': np.random.normal(2000, 500, n_samples).astype(int),
        'lot_size': np.random.exponential(8000, n_samples).astype(int),
        'year_built': np.random.choice(range(1950, 2023), n_samples),
        'garage': np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.6, 0.2]),
        'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples, p=[0.3, 0.6, 0.1]),
    }
    
    price = (
        house_data['sqft'] * 150 +
        house_data['bedrooms'] * 10000 +
        house_data['bathrooms'] * 8000 +
        house_data['garage'] * 5000 +
        np.random.normal(0, 20000, n_samples)
    )
    house_data['price'] = np.maximum(price, 50000).astype(int)
    
    pd.DataFrame(house_data).to_csv('data/houses.csv', index=False)
    print("✅ Created data/houses.csv")
    
    # Create sample config files
    classification_config = {
        "dataset": {
            "path": "data/customers.csv",
            "target_column": "churn"
        },
        "problem_type": "classification",
        "preprocessing": {
            "impute_missing": "mean",
            "scale_features": True
        },
        "model": {
            "auto_select": True,
            "options": ["logistic_regression", "random_forest", "xgboost"]
        },
        "output": {
            "save_dir": "results/classification/",
            "format": "pickle"
        }
    }
    
    regression_config = {
        "dataset": {
            "path": "data/houses.csv",
            "target_column": "price"
        },
        "problem_type": "regression",
        "preprocessing": {
            "impute_missing": "median",
            "scale_features": True
        },
        "model": {
            "auto_select": True,
            "options": ["linear_regression", "random_forest", "xgboost"]
        },
        "output": {
            "save_dir": "results/regression/",
            "format": "pickle"
        }
    }
    
    with open('configs/classification_example.json', 'w') as f:
        json.dump(classification_config, f, indent=2)
    print("✅ Created configs/classification_example.json")
    
    with open('configs/regression_example.json', 'w') as f:
        json.dump(regression_config, f, indent=2)
    print("✅ Created configs/regression_example.json")
    
    # Create README
    readme = '''# 🎮 RetroML - 90s-Style Automated Machine Learning

Welcome to RetroML! A nostalgic, configuration-driven ML pipeline that brings the radical vibes of the 90s to modern machine learning.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a Pipeline**
   ```bash
   python retroml.py configs/classification_example.json
   ```

3. **Try the Interactive UI** (if you have the UI file)
   ```bash
   python retroml_ui.py
   ```

## 📁 Project Structure

```
retroml/
├── retroml.py              # Main pipeline script
├── retroml_ui.py          # Interactive terminal UI (optional)
├── requirements.txt       # Python dependencies
├── configs/              # Configuration files
│   ├── classification_example.json
│   └── regression_example.json
├── data/                 # Sample datasets
│   ├── customers.csv     # Customer churn data
│   └── houses.csv        # House price data
└── results/              # Output directory
    ├── classification/
    └── regression/
```

## 🎯 Configuration Format

```json
{
  "dataset": {
    "path": "data/your_data.csv",
    "target_column": "target"
  },
  "problem_type": "classification",  // or "regression"
  "preprocessing": {
    "impute_missing": "mean",        // "mean", "median", "mode"
    "scale_features": true
  },
  "model": {
    "auto_select": true,
    "options": ["logistic_regression", "random_forest", "xgboost"]
  },
  "output": {
    "save_dir": "results/",
    "format": "pickle"
  }
}
```

## 🎮 Features

- ✅ **Automated model selection** - Tests multiple algorithms automatically
- ✅ **Config-driven workflow** - No coding required, just JSON configuration
- ✅ **Retro 90s UI** - ASCII art, colored output, loading animations
- ✅ **Comprehensive reporting** - Detailed evaluation reports and metrics
- ✅ **Model persistence** - Save and load trained models
- ✅ **Extensible architecture** - Easy to add new algorithms and preprocessing steps

## 🔧 Supported Algorithms

**Classification:**
- Logistic Regression
- Random Forest
- XGBoost (if installed)

**Regression:**
- Linear Regression  
- Random Forest
- XGBoost (if installed)

## 🎨 The Retro Experience

RetroML brings back the radical vibes of 90s computing:
- 🎮 ASCII art banners and loading screens
- 🌈 Colorful terminal output with Rich formatting
- 📼 Progress bars that look like old-school installers
- 💾 Nostalgic terminology and messaging

## 🚀 Example Usage

```bash
# Run classification pipeline
python retroml.py configs/classification_example.json

# Run regression pipeline  
python retroml.py configs/regression_example.json

# Create your own config and run
python retroml.py configs/my_config.json
```

## 📊 Output Files

After running a pipeline, you'll find:
- `best_model.pkl` - The trained model (with preprocessing)
- `evaluation_report.txt` - Detailed performance metrics
- Console output with colorful retro styling

## 🤖 Extending RetroML

Want to add new algorithms or preprocessing steps? The modular design makes it easy:

1. Add new models to `_get_default_models()`
2. Extend preprocessing options in `preprocess_data()`
3. Add new problem types (clustering, NLP, etc.)

## 🎯 Coming Soon (Roadmap)

- 🧮 Clustering algorithms (K-means, DBSCAN)
- 📝 NLP support with transformers
- 🔍 Hyperparameter tuning with Optuna
- 📊 Model explainability with SHAP
- 🚀 MLflow experiment tracking
- 🎛️ Web-based UI with Flask/Streamlit

---

*"Dude, this ML pipeline is totally tubular!" - RetroML, probably*

## 🐛 Troubleshooting

**ModuleNotFoundError**: Install requirements with `pip install -r requirements.txt`

**FileNotFoundError**: Make sure your dataset path in the config is correct

**Performance Issues**: Try reducing dataset size or using simpler models for large datasets

## 📄 License

This project is open source. Feel free to modify and distribute!

---

🎮 **RetroML** - Bringing 90s vibes to modern ML! 🎮
'''
    
    with open('README.md', 'w') as f:
        f.write(readme)
    print("✅ Created README.md")
    
    # Create simple test
    test_code = '''#!/usr/bin/env python3
"""
Simple test for RetroML
"""

import unittest
import json
import os
from pathlib import Path

class TestRetroML(unittest.TestCase):
    
    def test_config_files_exist(self):
        """Test that example config files exist"""
        self.assertTrue(Path('configs/classification_example.json').exists())
        self.assertTrue(Path('configs/regression_example.json').exists())
    
    def test_data_files_exist(self):
        """Test that sample data files exist"""
        self.assertTrue(Path('data/customers.csv').exists())
        self.assertTrue(Path('data/houses.csv').exists())
    
    def test_config_format(self):
        """Test that config files have correct format"""
        with open('configs/classification_example.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['dataset', 'problem_type', 'preprocessing', 'model', 'output']
        for key in required_keys:
            self.assertIn(key, config)

if __name__ == '__main__':
    unittest.main()
'''
    
    with open('tests/test_retroml.py', 'w') as f:
        f.write(test_code)
    print("✅ Created tests/test_retroml.py")
    
    # Create run script
    run_script = '''#!/usr/bin/env python3
"""
RetroML Runner Script
====================
Convenience script to run RetroML with different configurations
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    print("🎮 RETROML RUNNER 🎮")
    print("=" * 40)
    
    # Check if config files exist
    configs = list(Path('configs').glob('*.json'))
    
    if not configs:
        print("❌ No configuration files found!")
        print("Run this script from the RetroML project directory.")
        return
    
    print("Available configurations:")
    for i, config in enumerate(configs):
        print(f"  {i + 1}. {config.name}")
    
    print(f"  {len(configs) + 1}. Run tests")
    print(f"  {len(configs) + 2}. Exit")
    
    try:
        choice = int(input("\\nSelect an option: ")) - 1
        
        if choice < len(configs):
            config_path = configs[choice]
            print(f"\\n🚀 Running RetroML with {config_path.name}...")
            subprocess.run([sys.executable, 'retroml.py', str(config_path)])
        elif choice == len(configs):
            print("\\n🧪 Running tests...")
            subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'])
        else:
            print("👋 Goodbye!")
            
    except (ValueError, IndexError, KeyboardInterrupt):
        print("\\n👋 Goodbye!")

if __name__ == '__main__':
    main()
'''
    
    with open('run.py', 'w') as f:
        f.write(run_script)
    print("✅ Created run.py")
    
    print("\\n🎉 RETROML PROJECT CREATED SUCCESSFULLY! 🎉")
    print("\\n📋 Next steps:")
    print("1. pip install -r requirements.txt")
    print("2. python retroml.py configs/classification_example.json")
    print("3. python run.py  # For interactive selection")
    print("\\n🎮 Welcome to the radical world of RetroML! 🚀")

if __name__ == "__main__":
    create_complete_project()