#!/usr/bin/env python3
"""

// Copyright (c) Agile Creative Labs Inc.
// Licensed under the MIT License.


RetroML Regression Demo - Real World Applications
================================================
Lightweight demo showcasing practical regression use cases:
- House Price Prediction
- Sales Forecasting  
- Performance Metrics Analysis
- Interactive Price Calculator
- Batch Property Valuation
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
import random

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.layout import Layout
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class RetroMLRegressionPredictor:
    """
    🎮 RetroML Regression Predictor - Real World Applications 🎮
    """
    
    def __init__(self, model_path: str):
        self.console = Console() if HAS_RICH else None
        self.model_path = model_path
        self.model_package = None
        self.model = None
        self.scaler = None
        self.config = None
        self.training_features = None
        self.use_case = self._detect_use_case()
        
        self._load_model()
    
    def _detect_use_case(self) -> str:
        """Detect the likely use case based on model path and features"""
        if "house" in self.model_path.lower() or "price" in self.model_path.lower():
            return "house_price"
        elif "sales" in self.model_path.lower():
            return "sales_forecast"
        else:
            return "general"  # Default case
    
    def _retro_print(self, message: str, style: str = "white"):
        """Print message in retro style"""
        if self.console:
            self.console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    def _print_regression_banner(self):
        """Display regression demo banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════╗
║  ██████╗ ███████╗ ██████╗ ██████╗ ███████╗███████╗███████╗██╗ ██████╗ ███╗   ██╗║
║  ██╔══██╗██╔════╝██╔════╝ ██╔══██╗██╔════╝██╔════╝██╔════╝██║██╔═══██╗████╗  ██║║
║  ██████╔╝█████╗  ██║  ███╗██████╔╝█████╗  ███████╗███████╗██║██║   ██║██╔██╗ ██║║
║  ██╔══██╗██╔══╝  ██║   ██║██╔══██╗██╔══╝  ╚════██║╚════██║██║██║   ██║██║╚██╗██║║
║  ██║  ██║███████╗╚██████╔╝██║  ██║███████╗███████║███████║██║╚██████╔╝██║ ╚████║║
║  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝║
║                                                                              ║
║              🏠 Real World Regression Applications 🏠                        ║
║                           v1.0 LIVE DEMO                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
        """
        if self.console:
            self.console.print(banner, style="bold magenta")
        else:
            print(banner)
#    
    def _load_model(self):
        """Load the regression model with robust error handling"""
        self._retro_print(f"🔄 LOADING REGRESSION MODEL FROM: {self.model_path}", "bold yellow")
    
        try:
            # Define a temporary config class for unpickling
            class RetroMLConfig:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
        
            # Temporarily add to __main__ for pickle
            import __main__
            __main__.RetroMLConfig = RetroMLConfig
        
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)  # Load into temporary variable
        
            # Clean up our temporary class
            del __main__.RetroMLConfig

            # Convert dict config to object if needed
            if isinstance(model_package.get('config'), dict):
                model_package['config'] = RetroMLConfig(**model_package['config'])
    
            self.model_package = model_package  # Now assign to instance
            self.model = model_package['model']
            self.scaler = model_package.get('scaler')
            self.config = model_package.get('config')
        
            # Extract training features
            self._extract_training_features()
        
            self._retro_print("✅ REGRESSION MODEL LOADED!", "bold green")
            self._retro_print(f"📊 Model Type: {type(self.model).__name__}", "cyan")
        
            if self.config:
                problem_type = getattr(self.config, 'problem_type', 'regression')
                dataset_path = getattr(self.config, 'dataset_path', 'unknown')
                self._retro_print(f"🎯 Problem Type: {problem_type}", "cyan")
                self._retro_print(f"📁 Original Dataset: {dataset_path}", "cyan")
        
            if self.training_features:
                self._retro_print(f"🔧 Features: {len(self.training_features)} total", "cyan")
            
        except Exception as e:
            self._retro_print(f"❌ ERROR LOADING MODEL: {str(e)}", "bold red")
            raise

    def x_load_model(self):
        """Load the regression model with robust error handling"""
        self._retro_print(f"🔄 LOADING REGRESSION MODEL FROM: {self.model_path}", "bold yellow")
        
        try:
            
            import __main__
            __main__.RetroMLConfig = RetroMLConfig    
            
            with open(self.model_path, 'rb') as f:
                self.model_package = pickle.load(f)

            del __main__.RetroMLConfig

            # Convert dict config to object if needed
            if isinstance(model_package.get('config'), dict):
                model_package['config'] = RetroMLConfig(**model_package['config'])
        
            self.model_package = model_package
            
            self.model = self.model_package['model']
            self.scaler = self.model_package.get('scaler')
            self.config = self.model_package.get('config')
            
            # Extract training features
            self._extract_training_features()
            
            self._retro_print("✅ REGRESSION MODEL LOADED!", "bold green")
            self._retro_print(f"📊 Model Type: {type(self.model).__name__}", "cyan")
            
            if self.config:
                problem_type = getattr(self.config, 'problem_type', 'regression')
                dataset_path = getattr(self.config, 'dataset_path', 'unknown')
                self._retro_print(f"🎯 Problem Type: {problem_type}", "cyan")
                self._retro_print(f"📁 Original Dataset: {dataset_path}", "cyan")
            
            if self.training_features:
                self._retro_print(f"🔧 Features: {len(self.training_features)} total", "cyan")
                
        except Exception as e:
            self._retro_print(f"❌ ERROR LOADING MODEL: {str(e)}", "bold red")
            raise
    
    def _extract_training_features(self):
        """Extract training feature names"""
        try:
            if self.scaler and hasattr(self.scaler, 'feature_names_in_'):
                self.training_features = list(self.scaler.feature_names_in_)
            elif hasattr(self.model, 'feature_names_in_'):
                self.training_features = list(self.model.feature_names_in_)
            else:
                self.training_features = None
        except Exception:
            self.training_features = None
    
    def _align_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Align input features with training features"""
        if self.training_features is None:
            return data
        
        aligned_data = pd.DataFrame()
        
        for feature in self.training_features:
            if feature in data.columns:
                aligned_data[feature] = data[feature]
            else:
                # Add missing feature with intelligent defaults
                default_value = self._get_feature_default(feature)
                aligned_data[feature] = default_value
                self._retro_print(f"   Added missing feature '{feature}' = {default_value}", "yellow")
        
        return aligned_data
    
    def _get_feature_default(self, feature_name: str) -> float:
        """Get intelligent default values for missing features"""
        feature_name_lower = feature_name.lower()
        
        # House price prediction defaults
        if 'area' in feature_name_lower or 'sqft' in feature_name_lower:
            return 1500.0  # Average house size
        elif 'bedroom' in feature_name_lower:
            return 3.0
        elif 'bathroom' in feature_name_lower:
            return 2.0
        elif 'age' in feature_name_lower:
            return 10.0  # 10 years old
        elif 'garage' in feature_name_lower:
            return 1.0
        elif 'lot' in feature_name_lower:
            return 0.25  # Quarter acre
        
        # Sales prediction defaults
        elif 'quantity' in feature_name_lower or 'units' in feature_name_lower:
            return 100.0
        elif 'price' in feature_name_lower:
            return 50.0
        elif 'discount' in feature_name_lower:
            return 0.1  # 10% discount
        
        # General defaults
        elif 'id' in feature_name_lower:
            return random.randint(1000, 9999)
        else:
            return 0.0
    
    def _preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data"""
        self._retro_print("🔧 PREPROCESSING INPUT DATA...", "bold magenta")
        
        # Align features
        data = self._align_features(data)
        
        # Handle categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_vals = data[col].unique()
            mapping = {val: i for i, val in enumerate(unique_vals)}
            data[col] = data[col].map(mapping)
        
        # Fill missing values
        data = data.fillna(data.mean())
        
        # Convert to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.fillna(0)
        
        # Scale if scaler available
        if self.scaler is not None:
            try:
                data_scaled = self.scaler.transform(data)
                data = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
                self._retro_print("📏 Features scaled", "yellow")
            except Exception as e:
                self._retro_print(f"⚠️  Scaling failed: {e}", "yellow")
        
        return data
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single prediction"""
        df = pd.DataFrame([input_data])
        df_processed = self._preprocess_input(df)
        
        prediction = self.model.predict(df_processed)[0]
        
        # Calculate confidence interval (approximation)
        confidence_interval = self._estimate_confidence_interval(prediction)
        
        return {
            'prediction': prediction,
            'confidence_low': confidence_interval[0],
            'confidence_high': confidence_interval[1],
            'input_data': input_data
        }
    
    def _estimate_confidence_interval(self, prediction: float, confidence: float = 0.95) -> tuple:
        """Estimate confidence interval for prediction"""
        # Simple approximation - in practice you'd use model-specific methods
        margin = abs(prediction) * 0.1  # 10% margin
        return (prediction - margin, prediction + margin)
    
    def predict_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Make batch predictions"""
        self._retro_print(f"🚀 MAKING BATCH PREDICTIONS FOR {len(input_data)} SAMPLES...", "bold blue")
        
        df_processed = self._preprocess_input(input_data.copy())
        predictions = self.model.predict(df_processed)
        
        results = input_data.copy()
        results['prediction'] = predictions
        
        # Add confidence intervals
        margins = np.abs(predictions) * 0.1
        results['confidence_low'] = predictions - margins
        results['confidence_high'] = predictions + margins
        
        return results

def run_house_price_demo(predictor):
    """Demo for house price prediction"""
    console = Console() if HAS_RICH else None
    
    def demo_print(message: str, style: str = "white"):
        if console:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    demo_print("🏠 HOUSE PRICE PREDICTION DEMO", "bold green")
    demo_print("="*50, "cyan")
    
    # Sample house data
    sample_houses = [
        {
            'name': 'Cozy Suburban Home',
            'data': {
                'area': 1800, 'bedrooms': 3, 'bathrooms': 2,
                'age': 5, 'garage': 2, 'lot_size': 0.3
            }
        },
        {
            'name': 'Luxury Downtown Condo',
            'data': {
                'area': 1200, 'bedrooms': 2, 'bathrooms': 2,
                'age': 2, 'garage': 1, 'lot_size': 0.0
            }
        },
        {
            'name': 'Family House with Pool',
            'data': {
                'area': 2500, 'bedrooms': 4, 'bathrooms': 3,
                'age': 8, 'garage': 2, 'lot_size': 0.5
            }
        }
    ]
    
    for house in sample_houses:
        demo_print(f"\n🏡 {house['name']}:", "bold yellow")
        
        if console:
            table = Table(title="Property Details", style="cyan")
            table.add_column("Feature", style="yellow")
            table.add_column("Value", style="green")
            
            for key, value in house['data'].items():
                display_key = key.replace('_', ' ').title()
                if 'area' in key.lower():
                    display_value = f"{value} sq ft"
                elif 'lot' in key.lower():
                    display_value = f"{value} acres"
                elif 'age' in key.lower():
                    display_value = f"{value} years"
                else:
                    display_value = str(value)
                
                table.add_row(display_key, display_value)
            
            console.print(table)
        
        # Make prediction
        try:
            result = predictor.predict_single(house['data'])
            
            price = result['prediction']
            low = result['confidence_low']
            high = result['confidence_high']
            
            demo_print(f"💰 Estimated Price: ${price:,.2f}", "bold green")
            demo_print(f"📊 Range: ${low:,.2f} - ${high:,.2f}", "cyan")
            
        except Exception as e:
            demo_print(f"❌ Prediction failed: {e}", "red")

def run_sales_forecast_demo(predictor):
    """Demo for sales forecasting"""
    console = Console() if HAS_RICH else None
    
    def demo_print(message: str, style: str = "white"):
        if console:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    demo_print("📈 SALES FORECASTING DEMO", "bold green")
    demo_print("="*50, "cyan")
    
    # Sample sales scenarios
    scenarios = [
        {
            'name': 'Premium Product Launch',
            'data': {
                'product_price': 199.99, 'marketing_spend': 50000,
                'discount_rate': 0.1, 'season': 1, 'competitor_count': 3
            }
        },
        {
            'name': 'Budget Product Campaign',
            'data': {
                'product_price': 29.99, 'marketing_spend': 10000,
                'discount_rate': 0.2, 'season': 2, 'competitor_count': 8
            }
        }
    ]
    
    for scenario in scenarios:
        demo_print(f"\n🎯 {scenario['name']}:", "bold yellow")
        
        try:
            result = predictor.predict_single(scenario['data'])
            
            sales = result['prediction']
            low = result['confidence_low']
            high = result['confidence_high']
            
            demo_print(f"📊 Predicted Sales: {sales:,.0f} units", "bold green")
            demo_print(f"📈 Range: {low:,.0f} - {high:,.0f} units", "cyan")
            
            # Calculate revenue
            price = scenario['data'].get('product_price', 50)
            revenue = sales * price
            demo_print(f"💰 Expected Revenue: ${revenue:,.2f}", "green")
            
        except Exception as e:
            demo_print(f"❌ Prediction failed: {e}", "red")

def run_interactive_calculator(predictor):
    """Interactive prediction calculator"""
    console = Console() if HAS_RICH else None
    
    def demo_print(message: str, style: str = "white"):
        if console:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    demo_print("🎮 INTERACTIVE PREDICTION CALCULATOR", "bold cyan")
    demo_print("Enter your own values to get predictions!", "yellow")
    demo_print("(Press Ctrl+C to exit)\n", "magenta")
    
    # Determine input fields based on training features
    if predictor.training_features:
        input_fields = [f for f in predictor.training_features if 'id' not in f.lower()][:6]  # Limit to 6 fields
    else:
        # Default fields for house prices
        input_fields = ['area', 'bedrooms', 'bathrooms', 'age', 'garage', 'lot_size']
    
    demo_print(f"📝 Please provide values for: {', '.join(input_fields)}", "cyan")
    
    while True:
        try:
            demo_print("\n🔢 Enter prediction inputs:", "bold green")
            
            input_data = {}
            for field in input_fields:
                prompt = f"{field.replace('_', ' ').title()}: "
                try:
                    value = float(input(prompt))
                    input_data[field] = value
                except ValueError:
                    demo_print(f"⚠️  Invalid input for {field}, using default", "yellow")
                    input_data[field] = predictor._get_feature_default(field)
            
            # Make prediction
            result = predictor.predict_single(input_data)
            
            prediction = result['prediction']
            low = result['confidence_low']
            high = result['confidence_high']
            
            demo_print(f"\n🎯 PREDICTION RESULT:", "bold cyan")
            demo_print(f"   Predicted Value: {prediction:,.2f}", "bold green")
            demo_print(f"   Confidence Range: {low:,.2f} - {high:,.2f}", "cyan")
            
            demo_print("\n" + "-"*50 + "\n", "white")
            
        except KeyboardInterrupt:
            demo_print("\n👋 Thanks for using the calculator!", "bold cyan")
            break
        except Exception as e:
            demo_print(f"❌ Error: {e}", "red")

def run_batch_analysis_demo(predictor):
    """Demo for batch analysis"""
    console = Console() if HAS_RICH else None
    
    def demo_print(message: str, style: str = "white"):
        if console:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    demo_print("📊 BATCH ANALYSIS DEMO", "bold green")
    demo_print("="*50, "cyan")
    
    # Generate sample batch data
    np.random.seed(42)
    n_samples = 5
    
    # Create realistic sample data
    batch_data = pd.DataFrame({
        'area': np.random.normal(2000, 500, n_samples).astype(int),
        'bedrooms': np.random.choice([2, 3, 4, 5], n_samples),
        'bathrooms': np.random.choice([1, 2, 3], n_samples),
        'age': np.random.uniform(1, 30, n_samples).astype(int),
        'garage': np.random.choice([0, 1, 2, 3], n_samples),
        'lot_size': np.random.uniform(0.1, 1.0, n_samples).round(2)
    })
    
    demo_print(f"🔄 Analyzing {len(batch_data)} properties...", "yellow")
    
    try:
        results = predictor.predict_batch(batch_data)
        
        if console:
            table = Table(title="Batch Prediction Results", style="cyan")
            table.add_column("Property", style="yellow")
            table.add_column("Area (sqft)", style="white")
            table.add_column("Bed/Bath", style="white")
            table.add_column("Age", style="white")
            table.add_column("Predicted Value", style="bold green")
            table.add_column("Range", style="cyan")
            
            for idx, row in results.iterrows():
                table.add_row(
                    f"Property {idx + 1}",
                    f"{row['area']:,}",
                    f"{row['bedrooms']}/{row['bathrooms']}",
                    f"{row['age']} years",
                    f"${row['prediction']:,.0f}",
                    f"${row['confidence_low']:,.0f} - ${row['confidence_high']:,.0f}"
                )
            
            console.print(table)
        else:
            print("\nBatch Results:")
            for idx, row in results.iterrows():
                print(f"Property {idx + 1}: ${row['prediction']:,.0f}")
        
        # Summary statistics
        demo_print(f"\n📈 SUMMARY STATISTICS:", "bold yellow")
        demo_print(f"   Average Prediction: ${results['prediction'].mean():,.2f}", "green")
        demo_print(f"   Highest Value: ${results['prediction'].max():,.2f}", "green")
        demo_print(f"   Lowest Value: ${results['prediction'].min():,.2f}", "green")
        demo_print(f"   Standard Deviation: ${results['prediction'].std():,.2f}", "cyan")
        
    except Exception as e:
        demo_print(f"❌ Batch analysis failed: {e}", "red")

def main():
    """Main regression demo function"""
    console = Console() if HAS_RICH else None
    
    def demo_print(message: str, style: str = "white"):
        if console:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    # Check for regression model
    model_path = "results/regression/best_model.pkl"
    
    if not Path(model_path).exists():
        demo_print("❌ REGRESSION MODEL NOT FOUND!", "bold red")
        demo_print("🔧 Please run a regression pipeline first:", "yellow")
        demo_print("   python retroml.py configs/regression_example.json", "cyan")
        demo_print("\n💡 Or check these locations:", "yellow")
        
        # Check other possible locations
        other_paths = [
            "results/best_model.pkl",
            "results/classification/best_model.pkl",
            "models/regression_model.pkl"
        ]
        
        for path in other_paths:
            if Path(path).exists():
                demo_print(f"   Found model at: {path}", "green")
                model_path = path
                break
        else:
            return
    
    try:
        # Load predictor
        predictor = RetroMLRegressionPredictor(model_path)
        predictor._print_regression_banner()
        
        demo_print("🎮 REGRESSION DEMO MENU", "bold cyan")
        demo_print("Choose your demo:", "yellow")
        demo_print("1. 🏠 House Price Prediction", "green")
        demo_print("2. 📈 Sales Forecasting", "green")
        demo_print("3. 🎮 Interactive Calculator", "green")
        demo_print("4. 📊 Batch Analysis", "green")
        demo_print("5. 🎯 All Demos (recommended)", "bold green")
        demo_print("6. Exit", "red")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            run_house_price_demo(predictor)
        elif choice == "2":
            run_sales_forecast_demo(predictor)
        elif choice == "3":
            run_interactive_calculator(predictor)
        elif choice == "4":
            run_batch_analysis_demo(predictor)
        elif choice == "5":
            demo_print("\n🎉 RUNNING ALL DEMOS!", "bold magenta")
            run_house_price_demo(predictor)
            demo_print("\n" + "="*60, "magenta")
            run_sales_forecast_demo(predictor)
            demo_print("\n" + "="*60, "magenta")
            run_batch_analysis_demo(predictor)
            demo_print("\n" + "="*60, "magenta")
            run_interactive_calculator(predictor)
        elif choice == "6":
            demo_print("👋 Goodbye!", "cyan")
        else:
            demo_print("Invalid choice. Running all demos...", "yellow")
            run_house_price_demo(predictor)
            run_sales_forecast_demo(predictor)
            run_batch_analysis_demo(predictor)
        
        demo_print("\n🎉 REGRESSION DEMO COMPLETE! 🎉", "bold green")
        demo_print("💡 Use RetroMLRegressionPredictor in your applications!", "yellow")
        
    except Exception as e:
        demo_print(f"💥 DEMO FAILED: {str(e)}", "bold red")
        import traceback
        demo_print(f"Debug info: {traceback.format_exc()}", "red")

if __name__ == "__main__":
    main()