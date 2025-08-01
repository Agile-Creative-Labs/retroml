#!/usr/bin/env python3
"""

// Copyright (c) Agile Creative Labs Inc.
// Licensed under the MIT License.

RetroML Model Analyzer & Visualizer
====================================
Analyze and visualize trained models from the results folder

Deep Model Analysis: Comprehensive performance evaluation
Visual Analytics:

Feature importance plots
Data distribution analysis
Correlation heatmaps
Confusion matrices
ROC curves


Automated Reporting: Generates detailed analysis reports
Retro Styling: Dark theme with plasma color schemes
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from retroml_config import RetroMLConfig
import warnings
warnings.filterwarnings('ignore')

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class RetroMLAnalyzer:
    """
    🎮 RetroML Model Analyzer - Deep Dive into Your Models 🎮
    """
    
    def __init__(self, results_dir: str = "results/classification"):
        self.console = Console() if HAS_RICH else None
        self.results_dir = Path(results_dir)
        self.model_package = None
        self.model = None
        self.config = None
        
        # Set up matplotlib for retro styling
        plt.style.use('dark_background')
        
    def _retro_print(self, message: str, style: str = "white"):
        """Print message in retro style"""
        if self.console:
            self.console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    def _print_analyzer_banner(self):
        """Display analyzer banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║  ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ███╗   ███╗██╗  ██╗   ║
║  ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ████╗ ████║╚██╗██╔╝   ║
║  ██╔████╔██║██║   ██║██║  ██║█████╗  ██║         ██╔████╔██║ ╚███╔╝    ║
║  ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║         ██║╚██╔╝██║ ██╔██╗    ║
║  ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗    ██║ ╚═╝ ██║██╔╝ ██╗   ║
║  ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝    ╚═╝     ╚═╝╚═╝  ╚═╝   ║
║                                                                       ║
║                🔍 Model Analysis & Visualization 🔍                   ║
╚═══════════════════════════════════════════════════════════════════════╝
        """
        if self.console:
            self.console.print(banner, style="bold magenta")
        else:
            print(banner)
    
    def load_model_and_data(self):
        """Load the trained model and original data"""
        model_path = self.results_dir / "best_model.pkl"
        
        if not model_path.exists():
            self._retro_print("❌ MODEL NOT FOUND!", "bold red")
            self._retro_print(f"Expected path: {model_path}", "red")
            self._retro_print("Run the classification pipeline first!", "yellow")
            return False
        
        self._retro_print(f"🔄 LOADING MODEL FROM: {model_path}", "bold yellow")
        
        try:
            with open(model_path, 'rb') as f:
                self.model_package = pickle.load(f)
            
            self.model = self.model_package['model']
            self.config = self.model_package.get('config')
            
            self._retro_print("✅ MODEL LOADED SUCCESSFULLY!", "bold green")
            
            # Load original data for analysis
            if self.config and hasattr(self.config, 'dataset_path'):
                data_path = Path(self.config.dataset_path)
                if data_path.exists():
                    self.original_data = pd.read_csv(data_path)
                    self._retro_print(f"📊 Original data loaded: {self.original_data.shape}", "cyan")
                else:
                    self._retro_print("⚠️  Original data file not found", "yellow")
                    self.original_data = None
            
            return True
            
        except Exception as e:
            self._retro_print(f"❌ ERROR LOADING MODEL: {str(e)}", "bold red")
            return False
    
    def analyze_model_performance(self):
        """Analyze model performance from evaluation report"""
        self._retro_print("📊 ANALYZING MODEL PERFORMANCE...", "bold cyan")
        
        report_path = self.results_dir / "evaluation_report.txt"
        
        if report_path.exists():
            with open(report_path, 'r') as f:
                report_content = f.read()
            
            self._retro_print("📋 Evaluation Report Found:", "bold green")
            
            if self.console:
                panel = Panel(report_content, title="Model Evaluation Report", style="cyan")
                self.console.print(panel)
            else:
                print("\n" + "="*60)
                print(report_content)
                print("="*60)
        else:
            self._retro_print("⚠️  Evaluation report not found", "yellow")
    
    def create_feature_importance_plot(self):
        """Create feature importance visualization"""
        if not hasattr(self.model, 'feature_importances_'):
            self._retro_print("⚠️  Model doesn't have feature importances", "yellow")
            return
        
        self._retro_print("📊 CREATING FEATURE IMPORTANCE PLOT...", "bold blue")
        
        # Get feature names (approximation based on common features)
        feature_names = [
            'age', 'tenure_months', 'monthly_charges', 'total_charges',
            'contract_type', 'payment_method', 'internet_service', 
            'online_security', 'tech_support', 'streaming_tv', 'streaming_movies'
        ]
        
        importances = self.model.feature_importances_
        # Only use as many feature names as we have importances
        feature_names = feature_names[:len(importances)]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Sort features by importance
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_importances = importances[sorted_indices]
        
        # Create bar plot with retro colors
        colors = plt.cm.plasma(np.linspace(0, 1, len(sorted_features)))
        bars = plt.bar(range(len(sorted_features)), sorted_importances, color=colors)
        
        plt.title('🎯 Feature Importance Analysis', fontsize=16, color='cyan', fontweight='bold')
        plt.xlabel('Features', fontsize=12, color='white')
        plt.ylabel('Importance Score', fontsize=12, color='white')
        
        # Rotate x-axis labels
        plt.xticks(range(len(sorted_features)), 
                  [name.replace('_', ' ').title() for name in sorted_features], 
                  rotation=45, ha='right', color='white')
        plt.yticks(color='white')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, sorted_importances)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom', color='white', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
        
        self._retro_print(f"💾 Feature importance plot saved: {plot_path}", "green")
    
    def create_data_distribution_plots(self):
        """Create data distribution visualizations"""
        if self.original_data is None:
            self._retro_print("⚠️  Original data not available for distribution plots", "yellow")
            return
        
        self._retro_print("📊 CREATING DATA DISTRIBUTION PLOTS...", "bold blue")
        
        # Get target column
        target_col = self.config.target_column if self.config else 'churn'
        
        if target_col not in self.original_data.columns:
            self._retro_print(f"⚠️  Target column '{target_col}' not found", "yellow")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎯 Data Distribution Analysis', fontsize=16, color='cyan', fontweight='bold')
        
        # Plot 1: Target distribution
        target_counts = self.original_data[target_col].value_counts()
        colors = ['#ff6b6b', '#4ecdc4']
        
        axes[0,0].pie(target_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                     colors=colors, textprops={'color': 'white'})
        axes[0,0].set_title('Target Distribution', color='white', fontweight='bold')
        
        # Plot 2: Age distribution by churn
        if 'age' in self.original_data.columns:
            for i, churn_val in enumerate([0, 1]):
                data_subset = self.original_data[self.original_data[target_col] == churn_val]['age']
                axes[0,1].hist(data_subset, alpha=0.7, label=f'Churn = {churn_val}', 
                              color=colors[i], bins=20)
            axes[0,1].set_title('Age Distribution by Churn', color='white', fontweight='bold')
            axes[0,1].set_xlabel('Age', color='white')
            axes[0,1].set_ylabel('Frequency', color='white')
            axes[0,1].legend()
            axes[0,1].tick_params(colors='white')
        
        # Plot 3: Monthly charges distribution
        if 'monthly_charges' in self.original_data.columns:
            for i, churn_val in enumerate([0, 1]):
                data_subset = self.original_data[self.original_data[target_col] == churn_val]['monthly_charges']
                axes[1,0].hist(data_subset, alpha=0.7, label=f'Churn = {churn_val}', 
                              color=colors[i], bins=20)
            axes[1,0].set_title('Monthly Charges Distribution by Churn', color='white', fontweight='bold')
            axes[1,0].set_xlabel('Monthly Charges ($)', color='white')
            axes[1,0].set_ylabel('Frequency', color='white')
            axes[1,0].legend()
            axes[1,0].tick_params(colors='white')
        
        # Plot 4: Tenure distribution
        if 'tenure_months' in self.original_data.columns:
            for i, churn_val in enumerate([0, 1]):
                data_subset = self.original_data[self.original_data[target_col] == churn_val]['tenure_months']
                axes[1,1].hist(data_subset, alpha=0.7, label=f'Churn = {churn_val}', 
                              color=colors[i], bins=20)
            axes[1,1].set_title('Tenure Distribution by Churn', color='white', fontweight='bold')
            axes[1,1].set_xlabel('Tenure (Months)', color='white')
            axes[1,1].set_ylabel('Frequency', color='white')
            axes[1,1].legend()
            axes[1,1].tick_params(colors='white')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "data_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
        
        self._retro_print(f"💾 Data distribution plots saved: {plot_path}", "green")
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap"""
        if self.original_data is None:
            self._retro_print("⚠️  Original data not available for correlation analysis", "yellow")
            return
        
        self._retro_print("🔥 CREATING CORRELATION HEATMAP...", "bold blue")
        
        # Select numeric columns
        numeric_data = self.original_data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            self._retro_print("⚠️  No numeric columns found for correlation", "yellow")
            return
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Use a retro color scheme
        cmap = plt.cm.plasma
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=cmap,
                             center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                             fmt='.2f', annot_kws={'color': 'white'})
        
        plt.title('🔥 Feature Correlation Matrix', fontsize=16, color='cyan', fontweight='bold', pad=20)
        
        # Style the plot
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', color='white')
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, color='white')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "correlation_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
        
        self._retro_print(f"💾 Correlation heatmap saved: {plot_path}", "green")
    
    def analyze_prediction_patterns(self):
        """Analyze prediction patterns on original data"""
        if self.original_data is None:
            self._retro_print("⚠️  Original data not available for prediction analysis", "yellow")
            return
        
        self._retro_print("🔍 ANALYZING PREDICTION PATTERNS...", "bold blue")
        
        # Prepare data for prediction (simple preprocessing)
        data_for_prediction = self.original_data.copy()
        target_col = self.config.target_column if self.config else 'churn'
        
        if target_col in data_for_prediction.columns:
            X = data_for_prediction.drop(target_col, axis=1)
            y_true = data_for_prediction[target_col]
        else:
            self._retro_print(f"⚠️  Target column '{target_col}' not found", "yellow")
            return
        
        # Simple preprocessing for categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        # Apply scaling if scaler exists
        scaler = self.model_package.get('scaler')
        if scaler is not None:
            X_scaled = scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            y_pred_proba = None
            
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(X)[:, 1]  # Probability of positive class
            
            # Create analysis table
            if self.console:
                analysis_table = Table(title="🎯 Prediction Analysis Summary", style="cyan")
                analysis_table.add_column("Metric", style="yellow")
                analysis_table.add_column("Value", style="green")
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')
                
                analysis_table.add_row("Accuracy", f"{accuracy:.3f}")
                analysis_table.add_row("Precision", f"{precision:.3f}")
                analysis_table.add_row("Recall", f"{recall:.3f}")
                analysis_table.add_row("F1-Score", f"{f1:.3f}")
                analysis_table.add_row("Total Predictions", str(len(y_pred)))
                analysis_table.add_row("Predicted Churners", str(np.sum(y_pred == 1)))
                analysis_table.add_row("Predicted Stayers", str(np.sum(y_pred == 0)))
                
                self.console.print(analysis_table)
            
            # Create confusion matrix plot
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', 
                       xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'],
                       annot_kws={'color': 'white'})
            plt.title('🎯 Confusion Matrix', fontsize=16, color='cyan', fontweight='bold')
            plt.xlabel('Predicted', color='white')
            plt.ylabel('Actual', color='white')
            plt.xticks(color='white')
            plt.yticks(color='white')
            
            # Save confusion matrix
            plot_path = self.results_dir / "confusion_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='black')
            plt.show()
            
            self._retro_print(f"💾 Confusion matrix saved: {plot_path}", "green")
            
            # Create ROC curve if probabilities available
            if y_pred_proba is not None:
                from sklearn.metrics import roc_curve, auc
                
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='cyan', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', color='white')
                plt.ylabel('True Positive Rate', color='white')
                plt.title('🎯 ROC Curve Analysis', fontsize=16, color='cyan', fontweight='bold')
                plt.legend(loc="lower right")
                plt.xticks(color='white')
                plt.yticks(color='white')
                plt.grid(True, alpha=0.3)
                
                # Save ROC curve
                plot_path = self.results_dir / "roc_curve.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='black')
                plt.show()
                
                self._retro_print(f"💾 ROC curve saved: {plot_path}", "green")
        
        except Exception as e:
            self._retro_print(f"❌ Error in prediction analysis: {str(e)}", "red")
    
    def generate_model_summary_report(self):
        """Generate comprehensive model summary report"""
        self._retro_print("📋 GENERATING COMPREHENSIVE MODEL REPORT...", "bold green")
        
        report_path = self.results_dir / "model_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("🎮 RETROML MODEL ANALYSIS REPORT 🎮\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: {type(self.model).__name__}\n")
            
            if self.config:
                f.write(f"Problem Type: {self.config.problem_type}\n")
                f.write(f"Dataset: {self.config.dataset_path}\n")
                f.write(f"Target Column: {self.config.target_column}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("📊 MODEL CHARACTERISTICS\n")
            f.write("=" * 80 + "\n")
            
            # Model parameters
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                f.write("Model Parameters:\n")
                for key, value in params.items():
                    f.write(f"  {key}: {value}\n")
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                f.write("\nFeature Importances:\n")
                feature_names = ['age', 'tenure_months', 'monthly_charges', 'total_charges',
                               'contract_type', 'payment_method', 'internet_service', 
                               'online_security', 'tech_support']
                importances = self.model.feature_importances_
                
                for i, importance in enumerate(importances):
                    if i < len(feature_names):
                        f.write(f"  {feature_names[i]}: {importance:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("🎯 GENERATED VISUALIZATIONS\n")
            f.write("=" * 80 + "\n")
            f.write("The following visualization files have been created:\n")
            f.write("  • feature_importance.png - Feature importance analysis\n")
            f.write("  • data_distributions.png - Data distribution plots\n")
            f.write("  • correlation_heatmap.png - Feature correlation matrix\n")
            f.write("  • confusion_matrix.png - Model performance matrix\n")
            f.write("  • roc_curve.png - ROC curve analysis\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("💡 INSIGHTS & RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n")
            
            if hasattr(self.model, 'feature_importances_'):
                top_feature_idx = np.argmax(self.model.feature_importances_)
                f.write(f"• Most important feature appears to be feature #{top_feature_idx}\n")
            
            f.write("• Review feature importance plots to understand key drivers\n")
            f.write("• Check correlation heatmap for multicollinearity issues\n")
            f.write("• Monitor ROC curve for model discrimination ability\n")
            f.write("• Consider feature engineering based on distribution plots\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("🚀 MODEL DEPLOYMENT READY!\n")
            f.write("Use the RetroMLPredictor class to make predictions in production.\n")
            f.write("=" * 80 + "\n")
        
        self._retro_print(f"📋 Comprehensive report saved: {report_path}", "bold green")
    
    def run_full_analysis(self):
        """Run complete model analysis"""
        self._print_analyzer_banner()
        
        if not self.load_model_and_data():
            return
        
        self._retro_print("\n🚀 STARTING COMPREHENSIVE MODEL ANALYSIS...", "bold cyan")
        
        try:
            # Run all analysis components
            analyses = [
                ("📊 Performance Analysis", self.analyze_model_performance),
                ("📈 Feature Importance", self.create_feature_importance_plot),
                ("📊 Data Distributions", self.create_data_distribution_plots),
                ("🔥 Correlation Analysis", self.create_correlation_heatmap),
                ("🎯 Prediction Patterns", self.analyze_prediction_patterns),
                ("📋 Summary Report", self.generate_model_summary_report)
            ]
            
            if self.console:
                for name, func in track(analyses, description="Running analyses..."):
                    self._retro_print(f"\n{name}", "bold blue")
                    func()
            else:
                for name, func in analyses:
                    self._retro_print(f"\n{name}", "bold blue")
                    func()
            
            self._retro_print("\n🎉 ANALYSIS COMPLETE - MODEL FULLY ANALYZED! 🎉", "bold green")
            self._retro_print("📁 Check the results directory for all generated files", "yellow")
            
        except Exception as e:
            self._retro_print(f"💥 ANALYSIS FAILED: {str(e)}", "bold red")
            import traceback
            traceback.print_exc()

def main():
    """Main analyzer function"""
    console = Console() if HAS_RICH else None
    
    def demo_print(message: str, style: str = "white"):
        if console:
            console.print(f"[{style}]{message}[/{style}]")
        else:
            print(f">>> {message}")
    
    demo_print("🎮 RETROML MODEL ANALYZER 🎮", "bold magenta")
    demo_print("Deep dive into your trained models!", "yellow")
    
    # Check if results directory exists
    results_dir = Path("results/classification")
    if not results_dir.exists():
        demo_print("❌ RESULTS DIRECTORY NOT FOUND!", "bold red")
        demo_print("Please run the classification pipeline first:", "yellow")
        demo_print("  python retroml.py configs/classification_example.json", "cyan")
        return
    
    analyzer = RetroMLAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()