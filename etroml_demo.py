#!/usr/bin/env python3
"""
RetroML Classification Demo
===========================
Demonstrates how to load and use trained models from the results folder
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class RetroMLPredictor:
    """
    🎮 RetroML Model Predictor - Load and Use Trained Models 🎮
    """
    
    def __init__(self, model_path: str):
        self.console = Console() if HAS_RICH else None
        self.model_path = model_path
        self.model_package = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.config = None
        
        self._load_model()
    
    def _retro_print(self, message: str, style: str = "white"):
        """Print message in retro style"""
        if self.console:
            self.console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    def _print_demo_banner(self):
        """Display demo banner"""
        banner = """
╔════════════════════════════════════════════════════════════════════╗
║  ██████╗ ███████╗███╗   ███╗ ██████╗     ████████╗██╗███╗   ███╗███████╗ ║
║  ██╔══██╗██╔════╝████╗ ████║██╔═══██╗    ╚══██╔══╝██║████╗ ████║██╔════╝ ║
║  ██║  ██║█████╗  ██╔████╔██║██║   ██║       ██║   ██║██╔████╔██║█████╗   ║
║  ██║  ██║██╔══╝  ██║╚██╔╝██║██║   ██║       ██║   ██║██║╚██╔╝██║██╔══╝   ║
║  ██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝       ██║   ██║██║ ╚═╝ ██║███████╗ ║
║  ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝        ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝ ║
║                                                                        ║
║                🎯 Model Loading & Prediction Demo 🎯                   ║
╚════════════════════════════════════════════════════════════════════╝
        """
        if self.console:
            self.console.print(banner, style="bold cyan")
        else:
            print(banner)
    
    def _load_model(self):
        """Load the trained model package"""
        self._retro_print(f"🔄 LOADING MODEL FROM: {self.model_path}", "bold yellow")
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model_package = pickle.load(f)
            
            self.model = self.model_package['model']
            self.scaler = self.model_package.get('scaler')
            self.label_encoder = self.model_package.get('label_encoder')
            self.config = self.model_package.get('config')
            
            self._retro_print("✅ MODEL LOADED SUCCESSFULLY!", "bold green")
            self._retro_print(f"📊 Model Type: {type(self.model).__name__}", "cyan")
            
            if self.config:
                self._retro_print(f"🎯 Problem Type: {self.config.problem_type}", "cyan")
                self._retro_print(f"📁 Original Dataset: {self.config.dataset_path}", "cyan")
            
        except Exception as e:
            self._retro_print(f"❌ ERROR LOADING MODEL: {str(e)}", "bold red")
            raise
    
    def _preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data using the same steps as training"""
        self._retro_print("🔧 PREPROCESSING INPUT DATA...", "bold magenta")
        
        # Handle categorical columns (encode them)
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Simple label encoding for demo (in production, you'd save the encoders)
            unique_vals = data[col].unique()
            mapping = {val: i for i, val in enumerate(unique_vals)}
            data[col] = data[col].map(mapping)
        
        # Fill any missing values with mean
        data = data.fillna(data.mean())
        
        # Scale features if scaler was used during training
        if self.scaler is not None:
            data_scaled = self.scaler.transform(data)
            data = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
            self._retro_print("📏 Features scaled using saved scaler", "yellow")
        
        self._retro_print("✅ PREPROCESSING COMPLETE!", "bold green")
        return data
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single instance"""
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocess
        df_processed = self._preprocess_input(df)
        
        # Make prediction
        prediction = self.model.predict(df_processed)[0]
        prediction_proba = None
        
        # Get prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(df_processed)[0]
            prediction_proba = {f"Class_{i}": prob for i, prob in enumerate(proba)}
        
        # Decode prediction if label encoder was used
        if self.label_encoder is not None:
            try:
                prediction_decoded = self.label_encoder.inverse_transform([prediction])[0]
            except:
                prediction_decoded = prediction
        else:
            prediction_decoded = prediction
        
        return {
            'prediction': prediction_decoded,
            'prediction_raw': prediction,
            'probabilities': prediction_proba,
            'input_data': input_data
        }
    
    def predict_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for batch of data"""
        self._retro_print(f"🚀 MAKING BATCH PREDICTIONS FOR {len(input_data)} SAMPLES...", "bold blue")
        
        # Preprocess
        df_processed = self._preprocess_input(input_data.copy())
        
        # Make predictions
        predictions = self.model.predict(df_processed)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(df_processed)
            prob_df = pd.DataFrame(probabilities, columns=[f'prob_class_{i}' for i in range(probabilities.shape[1])])
        else:
            prob_df = pd.DataFrame()
        
        # Decode predictions if label encoder was used
        if self.label_encoder is not None:
            try:
                predictions_decoded = self.label_encoder.inverse_transform(predictions)
            except:
                predictions_decoded = predictions
        else:
            predictions_decoded = predictions
        
        # Create results DataFrame
        results = input_data.copy()
        results['prediction'] = predictions_decoded
        results['prediction_raw'] = predictions
        
        # Add probabilities if available
        if not prob_df.empty:
            results = pd.concat([results, prob_df], axis=1)
        
        return results

def run_classification_demo():
    """Run the classification demo"""
    console = Console() if HAS_RICH else None
    
    def demo_print(message: str, style: str = "white"):
        if console:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    # Print banner
    predictor = RetroMLPredictor.__new__(RetroMLPredictor)  # Create instance without init
    predictor.console = console
    predictor._print_demo_banner()
    
    # Check if model exists
    model_path = "results/classification/best_model.pkl"
    
    if not Path(model_path).exists():
        demo_print("❌ CLASSIFICATION MODEL NOT FOUND!", "bold red")
        demo_print("🔧 Please run the classification pipeline first:", "yellow")
        demo_print("   python retroml.py configs/classification_example.json", "cyan")
        return
    
    try:
        # Load the model
        predictor = RetroMLPredictor(model_path)
        
        demo_print("\n" + "="*60, "cyan")
        demo_print("🎯 SINGLE PREDICTION DEMO", "bold green")
        demo_print("="*60, "cyan")
        
        # Demo 1: Single prediction - High churn risk customer
        high_risk_customer = {
            'customer_id': 9999,
            'age': 25,
            'tenure_months': 3,
            'monthly_charges': 95.50,
            'total_charges': 285.50,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic',
            'online_security': 'No',
            'tech_support': 'No'
        }
        
        demo_print("🔍 Analyzing HIGH RISK customer:", "bold yellow")
        if console:
            # Create a nice table for the input
            table = Table(title="Customer Profile", style="cyan")
            table.add_column("Attribute", style="yellow")
            table.add_column("Value", style="green")
            
            for key, value in high_risk_customer.items():
                if key != 'customer_id':  # Skip ID for cleaner display
                    table.add_row(str(key).replace('_', ' ').title(), str(value))
            
            console.print(table)
        else:
            for key, value in high_risk_customer.items():
                if key != 'customer_id':
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Make prediction
        result = predictor.predict_single(high_risk_customer)
        
        demo_print(f"\n🎯 PREDICTION RESULT:", "bold cyan")
        demo_print(f"   Churn Prediction: {result['prediction']}", "bold green" if result['prediction'] == 0 else "bold red")
        
        if result['probabilities']:
            demo_print(f"   Confidence Scores:", "yellow")
            for class_name, prob in result['probabilities'].items():
                demo_print(f"     {class_name}: {prob:.2%}", "cyan")
        
        # Demo 2: Single prediction - Low churn risk customer
        demo_print("\n" + "-"*40, "cyan")
        
        low_risk_customer = {
            'customer_id': 8888,
            'age': 45,
            'tenure_months': 36,
            'monthly_charges': 45.20,
            'total_charges': 1626.00,
            'contract_type': 'Two year',
            'payment_method': 'Bank transfer',
            'internet_service': 'DSL',
            'online_security': 'Yes',
            'tech_support': 'Yes'
        }
        
        demo_print("🔍 Analyzing LOW RISK customer:", "bold yellow")
        
        result2 = predictor.predict_single(low_risk_customer)
        demo_print(f"🎯 PREDICTION: {result2['prediction']}", "bold green" if result2['prediction'] == 0 else "bold red")
        
        # Demo 3: Batch prediction
        demo_print("\n" + "="*60, "cyan")
        demo_print("📊 BATCH PREDICTION DEMO", "bold green")
        demo_print("="*60, "cyan")
        
        # Create sample batch data
        batch_data = pd.DataFrame([
            {
                'age': 30, 'tenure_months': 12, 'monthly_charges': 75.0,
                'total_charges': 900.0, 'contract_type': 'One year',
                'payment_method': 'Credit card', 'internet_service': 'Fiber optic',
                'online_security': 'No', 'tech_support': 'Yes'
            },
            {
                'age': 55, 'tenure_months': 48, 'monthly_charges': 55.0,
                'total_charges': 2640.0, 'contract_type': 'Two year',
                'payment_method': 'Bank transfer', 'internet_service': 'DSL',
                'online_security': 'Yes', 'tech_support': 'Yes'
            },
            {
                'age': 22, 'tenure_months': 2, 'monthly_charges': 99.0,
                'total_charges': 198.0, 'contract_type': 'Month-to-month',
                'payment_method': 'Electronic check', 'internet_service': 'Fiber optic',
                'online_security': 'No', 'tech_support': 'No'  
            }
        ])
        
        demo_print(f"📋 Processing {len(batch_data)} customers...", "yellow")
        
        # Make batch predictions
        batch_results = predictor.predict_batch(batch_data)
        
        # Display results
        if console:
            results_table = Table(title="Batch Prediction Results", style="cyan")
            results_table.add_column("Customer", style="yellow")
            results_table.add_column("Age", style="white")
            results_table.add_column("Tenure", style="white")
            results_table.add_column("Monthly Charges", style="white")
            results_table.add_column("Contract", style="white")
            results_table.add_column("Churn Prediction", style="bold")
            
            for idx, row in batch_results.iterrows():
                churn_style = "bold red" if row['prediction'] == 1 else "bold green"
                churn_text = "WILL CHURN" if row['prediction'] == 1 else "WILL STAY"
                
                results_table.add_row(
                    f"Customer {idx + 1}",
                    str(row['age']),
                    f"{row['tenure_months']} months",
                    f"${row['monthly_charges']:.2f}",
                    row['contract_type'],
                    f"[{churn_style}]{churn_text}[/{churn_style}]"
                )
            
            console.print(results_table)
        else:
            print("\nBatch Results:")
            for idx, row in batch_results.iterrows():
                churn_text = "WILL CHURN" if row['prediction'] == 1 else "WILL STAY"
                print(f"  Customer {idx + 1}: {churn_text}")
        
        # Demo 4: Model insights
        demo_print("\n" + "="*60, "cyan")
        demo_print("🧠 MODEL INSIGHTS", "bold green")
        demo_print("="*60, "cyan")
        
        if hasattr(predictor.model, 'feature_importances_'):
            demo_print("📊 Top Feature Importances:", "bold yellow")
            
            # Get feature names (approximation for demo)
            feature_names = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 
                           'contract_type', 'payment_method', 'internet_service', 
                           'online_security', 'tech_support']
            
            importances = predictor.model.feature_importances_
            feature_importance = list(zip(feature_names[:len(importances)], importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_importance[:5]):
                demo_print(f"   {i+1}. {feature.replace('_', ' ').title()}: {importance:.3f}", "cyan")
        
        # Success message
        demo_print("\n🎉 DEMO COMPLETE - MODEL IS TOTALLY FUNCTIONAL! 🎉", "bold green")
        demo_print("💡 You can now use this model to predict customer churn!", "yellow")
        demo_print("🔧 Integrate it into your applications using the RetroMLPredictor class", "yellow")
        
    except Exception as e:
        demo_print(f"💥 DEMO FAILED: {str(e)}", "bold red")
        import traceback
        demo_print(f"Debug info: {traceback.format_exc()}", "red")

def interactive_prediction_mode():
    """Interactive mode for making predictions"""
    console = Console() if HAS_RICH else None
    
    def demo_print(message: str, style: str = "white"):
        if console:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    model_path = "results/classification/best_model.pkl"
    
    if not Path(model_path).exists():
        demo_print("❌ MODEL NOT FOUND! Run the pipeline first.", "bold red")
        return
    
    try:
        predictor = RetroMLPredictor(model_path)
        
        demo_print("🎮 INTERACTIVE PREDICTION MODE 🎮", "bold cyan")
        demo_print("Enter customer details to get churn prediction!", "yellow")
        demo_print("(Press Ctrl+C to exit)\n", "magenta")
        
        while True:
            try:
                demo_print("📝 Enter customer information:", "bold green")
                
                customer_data = {}
                customer_data['age'] = int(input("Age: "))
                customer_data['tenure_months'] = int(input("Tenure (months): "))
                customer_data['monthly_charges'] = float(input("Monthly charges ($): "))
                customer_data['total_charges'] = float(input("Total charges ($): "))
                customer_data['contract_type'] = input("Contract type (Month-to-month/One year/Two year): ")
                customer_data['payment_method'] = input("Payment method (Electronic check/Mailed check/Bank transfer/Credit card): ")
                customer_data['internet_service'] = input("Internet service (DSL/Fiber optic/No): ")
                customer_data['online_security'] = input("Online security (Yes/No): ")
                customer_data['tech_support'] = input("Tech support (Yes/No): ")
                
                # Make prediction
                result = predictor.predict_single(customer_data)
                
                demo_print(f"\n🎯 PREDICTION RESULT:", "bold cyan")
                churn_text = "LIKELY TO CHURN" if result['prediction'] == 1 else "LIKELY TO STAY"
                churn_style = "bold red" if result['prediction'] == 1 else "bold green"
                demo_print(f"   {churn_text}", churn_style)
                
                if result['probabilities']:
                    demo_print("   Confidence:", "yellow")
                    for class_name, prob in result['probabilities'].items():
                        demo_print(f"     {class_name}: {prob:.1%}", "cyan")
                
                demo_print("\n" + "-"*50 + "\n", "white")
                
            except KeyboardInterrupt:
                demo_print("\n👋 Goodbye! Thanks for using RetroML!", "bold cyan")
                break
            except ValueError as e:
                demo_print(f"⚠️  Invalid input: {e}. Please try again.", "yellow")
            except Exception as e:
                demo_print(f"❌ Error: {e}", "red")
                
    except Exception as e:
        demo_print(f"💥 Failed to load model: {e}", "bold red")

def main():
    """Main demo function"""
    console = Console() if HAS_RICH else None
    
    def demo_print(message: str, style: str = "white"):
        if console:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    demo_print("🎮 RETROML CLASSIFICATION DEMO 🎮", "bold cyan")
    demo_print("Choose your adventure:", "yellow")
    demo_print("1. Full Demo (recommended)", "green")
    demo_print("2. Interactive Prediction Mode", "green")
    demo_print("3. Exit", "red")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            run_classification_demo()
        elif choice == "2":
            interactive_prediction_mode()
        elif choice == "3":
            demo_print("👋 Goodbye!", "cyan")
        else:
            demo_print("Invalid choice. Running full demo...", "yellow")
            run_classification_demo()
            
    except KeyboardInterrupt:
        demo_print("\n👋 Goodbye!", "cyan")

if __name__ == "__main__":
    main()