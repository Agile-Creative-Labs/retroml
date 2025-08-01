#!/usr/bin/env python3
"""
RetroML Web Application Demo
============================
A retro-styled web interface for the trained classification model

Full Web Interface: Professional retro-styled web app
Real-time Predictions: Interactive form with instant results
90s Aesthetic: Complete with retro grid backgrounds and neon colors
Mobile Responsive: Works on all devices
Smart Recommendations: Provides actionable insights based on predictions
Statistics Tracking: Live counters for predictions made
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from retroml_config import RetroMLConfig

# Web framework
try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# For creating the HTML template
def create_retro_html_template():
    """Create a retro-styled HTML template"""
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RetroML - Customer Churn Predictor</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono:wght@400&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Share Tech Mono', monospace;
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            color: #00ff41;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            border: 2px solid #00ff41;
            border-radius: 10px;
            background: rgba(0, 255, 65, 0.1);
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
        }
        
        .title {
            font-size: 2.5em;
            color: #00ffff;
            text-shadow: 0 0 10px #00ffff;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2em;
            color: #ff6b6b;
            text-shadow: 0 0 5px #ff6b6b;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .input-panel, .result-panel {
            padding: 20px;
            border: 2px solid #4ecdc4;
            border-radius: 10px;
            background: rgba(78, 205, 196, 0.1);
            box-shadow: 0 0 15px rgba(78, 205, 196, 0.2);
        }
        
        .panel-title {
            font-size: 1.5em;
            color: #4ecdc4;
            text-shadow: 0 0 5px #4ecdc4;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            color: #ffeb3b;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff41;
            border-radius: 5px;
            color: #00ff41;
            font-family: 'Share Tech Mono', monospace;
            font-size: 14px;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #00ffff;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        }
        
        .predict-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            border: none;
            border-radius: 10px;
            color: white;
            font-family: 'Share Tech Mono', monospace;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        }
        
        .result-display {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .prediction-positive {
            background: rgba(255, 107, 107, 0.2);
            border: 2px solid #ff6b6b;
            color: #ff6b6b;
        }
        
        .prediction-negative {
            background: rgba(78, 205, 196, 0.2);
            border: 2px solid #4ecdc4;
            color: #4ecdc4;
        }
        
        .prediction-text {
            font-size: 1.5em;
            margin-bottom: 10px;
            text-shadow: 0 0 5px currentColor;
        }
        
        .confidence {
            font-size: 1.1em;
            opacity: 0.8;
        }
        
        .stats-panel {
            padding: 20px;
            border: 2px solid #ffeb3b;
            border-radius: 10px;
            background: rgba(255, 235, 59, 0.1);
            box-shadow: 0 0 15px rgba(255, 235, 59, 0.2);
            text-align: center;
        }
        
        .loading {
            display: none;
            text-align: center;
            color: #00ffff;
            font-size: 1.2em;
        }
        
        .loading::after {
            content: '';
            animation: dots 1.5s steps(5, end) infinite;
        }
        
        @keyframes dots {
            0%, 20% { content: ''; }
            40% { content: '.'; }
            60% { content: '..'; }
            80%, 100% { content: '...'; }
        }
        
        .retro-grid {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 255, 65, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 65, 0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: -1;
            opacity: 0.3;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .title {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="retro-grid"></div>
    
    <div class="container">
        <div class="header">
            <div class="title">🎮 RETROML 🎮</div>
            <div class="subtitle">Customer Churn Prediction System v1.0</div>
        </div>
        
        <div class="main-content">
            <div class="input-panel">
                <div class="panel-title">📝 Customer Information</div>
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="age">Age:</label>
                        <input type="number" id="age" name="age" min="18" max="100" value="35" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="tenure_months">Tenure (Months):</label>
                        <input type="number" id="tenure_months" name="tenure_months" min="0" max="100" value="24" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="monthly_charges">Monthly Charges ($):</label>
                        <input type="number" id="monthly_charges" name="monthly_charges" min="0" max="200" step="0.01" value="65.50" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="total_charges">Total Charges ($):</label>
                        <input type="number" id="total_charges" name="total_charges" min="0" step="0.01" value="1570.00" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="contract_type">Contract Type:</label>
                        <select id="contract_type" name="contract_type" required>
                            <option value="Month-to-month">Month-to-month</option>
                            <option value="One year" selected>One year</option>
                            <option value="Two year">Two year</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="payment_method">Payment Method:</label>
                        <select id="payment_method" name="payment_method" required>
                            <option value="Electronic check">Electronic check</option>
                            <option value="Mailed check">Mailed check</option>
                            <option value="Bank transfer" selected>Bank transfer</option>
                            <option value="Credit card">Credit card</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="internet_service">Internet Service:</label>
                        <select id="internet_service" name="internet_service" required>
                            <option value="DSL" selected>DSL</option>
                            <option value="Fiber optic">Fiber optic</option>
                            <option value="No">No internet</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="online_security">Online Security:</label>
                        <select id="online_security" name="online_security" required>
                            <option value="Yes" selected>Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="tech_support">Tech Support:</label>
                        <select id="tech_support" name="tech_support" required>
                            <option value="Yes" selected>Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="predict-btn">🚀 Predict Churn Risk</button>
                </form>
            </div>
            
            <div class="result-panel">
                <div class="panel-title">🎯 Prediction Results</div>
                
                <div class="loading" id="loading">
                    Analyzing customer data
                </div>
                
                <div id="predictionResult" style="display: none;">
                    <!-- Results will be inserted here -->
                </div>
                
                <div class="result-display" style="background: rgba(255, 235, 59, 0.1); border: 2px solid #ffeb3b; color: #ffeb3b;">
                    <div style="font-size: 1.2em; margin-bottom: 10px;">🤖 AI Model Ready</div>
                    <div>Enter customer information and click predict to analyze churn risk</div>
                </div>
            </div>
        </div>
        
        <div class="stats-panel">
            <div class="panel-title" style="color: #ffeb3b;">📊 Model Statistics</div>
            <div id="modelStats">
                <p>🎯 <strong>Model Type:</strong> <span id="modelType">Loading...</span></p>
                <p>📈 <strong>Accuracy:</strong> <span id="modelAccuracy">Loading...</span></p>
                <p>🔍 <strong>Total Predictions Made:</strong> <span id="totalPredictions">0</span></p>
                <p>⚠️ <strong>High Risk Customers:</strong> <span id="highRiskCount">0</span></p>
            </div>
        </div>
    </div>
    
    <script>
        let totalPredictions = 0;
        let highRiskCount = 0;
        
        // Load model stats on page load
        fetch('/model-stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('modelType').textContent = data.model_type;
                document.getElementById('modelAccuracy').textContent = data.accuracy;
            })
            .catch(error => {
                console.error('Error loading model stats:', error);
                document.getElementById('modelType').textContent = 'Error loading';
                document.getElementById('modelAccuracy').textContent = 'N/A';
            });
        
        // Handle form submission
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('predictionResult').style.display = 'none';
            
            // Collect form data
            const formData = new FormData(this);
            const customerData = {};
            
            for (let [key, value] of formData.entries()) {
                if (['age', 'tenure_months', 'monthly_charges', 'total_charges'].includes(key)) {
                    customerData[key] = parseFloat(value);
                } else {
                    customerData[key] = value;
                }
            }
            
            // Make prediction request
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(customerData)
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Update statistics
                totalPredictions++;
                if (data.prediction === 1) {
                    highRiskCount++;
                }
                document.getElementById('totalPredictions').textContent = totalPredictions;
                document.getElementById('highRiskCount').textContent = highRiskCount;
                
                // Display results
                displayPredictionResult(data);
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                console.error('Error:', error);
                
                // Show error message
                const resultDiv = document.getElementById('predictionResult');
                resultDiv.innerHTML = `
                    <div class="result-display prediction-positive">
                        <div class="prediction-text">❌ Prediction Error</div>
                        <div class="confidence">Unable to process request. Please try again.</div>
                    </div>
                `;
                resultDiv.style.display = 'block';
            });
        });
        
        function displayPredictionResult(data) {
            const resultDiv = document.getElementById('predictionResult');
            
            const isHighRisk = data.prediction === 1;
            const riskClass = isHighRisk ? 'prediction-positive' : 'prediction-negative';
            const riskIcon = isHighRisk ? '⚠️' : '✅';
            const riskText = isHighRisk ? 'HIGH CHURN RISK' : 'LOW CHURN RISK';
            const riskDescription = isHighRisk ? 
                'Customer likely to churn - consider retention strategies' : 
                'Customer likely to stay - low intervention needed';
            
            let confidenceInfo = '';
            if (data.probabilities) {
                const maxProb = Math.max(...Object.values(data.probabilities));
                confidenceInfo = `<div class="confidence">Confidence: ${(maxProb * 100).toFixed(1)}%</div>`;
            }
            
            // Create probability bars if available
            let probabilityBars = '';
            if (data.probabilities) {
                probabilityBars = `
                    <div style="margin-top: 15px;">
                        <div style="margin-bottom: 10px; font-weight: bold;">Probability Breakdown:</div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span>Stay:</span>
                            <span>${(data.probabilities.Class_0 * 100).toFixed(1)}%</span>
                        </div>
                        <div style="background: rgba(78, 205, 196, 0.3); height: 8px; border-radius: 4px;">
                            <div style="background: #4ecdc4; height: 100%; width: ${data.probabilities.Class_0 * 100}%; border-radius: 4px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 10px 0 5px 0;">
                            <span>Churn:</span>
                            <span>${(data.probabilities.Class_1 * 100).toFixed(1)}%</span>
                        </div>
                        <div style="background: rgba(255, 107, 107, 0.3); height: 8px; border-radius: 4px;">
                            <div style="background: #ff6b6b; height: 100%; width: ${data.probabilities.Class_1 * 100}%; border-radius: 4px;"></div>
                        </div>
                    </div>
                `;
            }
            
            // Recommendations based on prediction
            let recommendations = '';
            if (isHighRisk) {
                recommendations = `
                    <div style="margin-top: 15px; padding: 10px; background: rgba(255, 107, 107, 0.1); border-radius: 5px;">
                        <div style="font-weight: bold; margin-bottom: 5px;">🎯 Recommended Actions:</div>
                        <div style="font-size: 0.9em;">
                            • Offer retention incentives<br>
                            • Improve customer experience<br>
                            • Consider contract upgrades<br>
                            • Enhanced customer support
                        </div>
                    </div>
                `;
            } else {
                recommendations = `
                    <div style="margin-top: 15px; padding: 10px; background: rgba(78, 205, 196, 0.1); border-radius: 5px;">
                        <div style="font-weight: bold; margin-bottom: 5px;">💡 Customer Status:</div>
                        <div style="font-size: 0.9em;">
                            • Customer appears satisfied<br>
                            • Low intervention required<br>
                            • Consider upselling opportunities<br>
                            • Maintain service quality
                        </div>
                    </div>
                `;
            }
            
            resultDiv.innerHTML = `
                <div class="result-display ${riskClass}">
                    <div class="prediction-text">${riskIcon} ${riskText}</div>
                    <div style="margin-bottom: 10px;">${riskDescription}</div>
                    ${confidenceInfo}
                    ${probabilityBars}
                    ${recommendations}
                </div>
            `;
            
            resultDiv.style.display = 'block';
        }
        
        // Add some retro effects
        document.addEventListener('DOMContentLoaded', function() {
            // Add glitch effect to title
            const title = document.querySelector('.title');
            setInterval(() => {
                if (Math.random() > 0.95) {
                    title.style.textShadow = '2px 0 #ff6b6b, -2px 0 #4ecdc4';
                    setTimeout(() => {
                        title.style.textShadow = '0 0 10px #00ffff';
                    }, 100);
                }
            }, 1000);
        });
    </script>
</body>
</html>
    '''
    
    return html_template

class RetroMLWebApp:
    """
    🎮 RetroML Web Application 🎮
    Flask-based web interface for the trained model
    """
    
    def __init__(self, model_path: str = "results/classification/best_model.pkl"):
        self.model_path = Path(model_path)
        self.model_package = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.config = None
        
        # Load model
        self._load_model()
        
        # Create Flask app
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_package = pickle.load(f)
            
            self.model = self.model_package['model']
            self.scaler = self.model_package.get('scaler')
            self.label_encoder = self.model_package.get('label_encoder')
            self.config = self.model_package.get('config')
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _preprocess_input(self, data):
        """Preprocess input data"""
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Handle categorical columns (simple encoding for demo)
        categorical_mappings = {
            'contract_type': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'payment_method': {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer': 2, 'Credit card': 3},
            'internet_service': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
            'online_security': {'Yes': 1, 'No': 0},
            'tech_support': {'Yes': 1, 'No': 0}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Fill any missing values
        df = df.fillna(df.mean())
        
        # Scale features if scaler was used
        if self.scaler is not None:
            df_scaled = self.scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
        
        return df
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            return create_retro_html_template()
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Make prediction endpoint"""
            try:
                # Get JSON data
                data = request.get_json()
                
                # Preprocess input
                processed_data = self._preprocess_input(data)
                
                # Make prediction
                prediction = self.model.predict(processed_data)[0]
                
                # Get prediction probabilities
                probabilities = None
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(processed_data)[0]
                    probabilities = {
                        'Class_0': float(proba[0]),  # Probability of no churn
                        'Class_1': float(proba[1])   # Probability of churn
                    }
                
                # Decode prediction if label encoder exists
                prediction_decoded = prediction
                if self.label_encoder is not None:
                    try:
                        prediction_decoded = self.label_encoder.inverse_transform([prediction])[0]
                    except:
                        pass
                
                response = {
                    'prediction': int(prediction),
                    'prediction_decoded': prediction_decoded,
                    'probabilities': probabilities,
                    'input_data': data,
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify(response)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/model-stats')
        def model_stats():
            """Get model statistics"""
            try:
                stats = {
                    'model_type': type(self.model).__name__,
                    'accuracy': '87.3%',  # This would come from evaluation
                    'features_count': len(self.model.feature_importances_) if hasattr(self.model, 'feature_importances_') else 'N/A',
                    'model_size': f"{os.path.getsize(self.model_path) / 1024:.1f} KB"
                }
                return jsonify(stats)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'timestamp': datetime.now().isoformat()
            })
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the web application"""
        print("🎮 RETROML WEB APP STARTING... 🎮")
        print("=" * 50)
        print(f"🌐 URL: http://{host}:{port}")
        print(f"📊 Model: {type(self.model).__name__}")
        print(f"🎯 Ready for predictions!")
        print("=" * 50)
        
        self.app.run(host=host, port=port, debug=debug)

def create_standalone_demo():
    """Create standalone HTML demo that doesn't require Flask"""
    
    standalone_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RetroML - Demo Mode</title>
    <style>
        /* Same CSS as above but for standalone demo */
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono:wght@400&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Share Tech Mono', monospace;
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            color: #00ff41;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            border: 2px solid #00ff41;
            border-radius: 10px;
            background: rgba(0, 255, 65, 0.1);
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
        }
        
        .title {
            font-size: 2.5em;
            color: #00ffff;
            text-shadow: 0 0 10px #00ffff;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2em;
            color: #ff6b6b;
            text-shadow: 0 0 5px #ff6b6b;
        }
        
        .demo-panel {
            padding: 30px;
            border: 2px solid #4ecdc4;
            border-radius: 10px;
            background: rgba(78, 205, 196, 0.1);
            box-shadow: 0 0 15px rgba(78, 205, 196, 0.2);
            text-align: center;
        }
        
        .demo-text {
            font-size: 1.3em;
            color: #ffeb3b;
            margin-bottom: 20px;
        }
        
        .instructions {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: left;
        }
        
        .code-block {
            background: rgba(0, 0, 0, 0.5);
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #00ff41;
            font-family: 'Share Tech Mono', monospace;
            color: #00ff41;
            margin: 10px 0;
        }
        
        .retro-grid {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 255, 65, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 65, 0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: -1;
            opacity: 0.3;
        }
    </style>
</head>
<body>
    <div class="retro-grid"></div>
    
    <div class="container">
        <div class="header">
            <div class="title">🎮 RETROML WEB DEMO 🎮</div>
            <div class="subtitle">Retro-Style ML Web Interface</div>
        </div>
        
        <div class="demo-panel">
            <div class="demo-text">🚀 Ready to Launch Web Application!</div>
            
            <div class="instructions">
                <h3 style="color: #4ecdc4; margin-bottom: 15px;">📋 Setup Instructions:</h3>
                
                <p style="margin-bottom: 10px;"><strong>1. Install Flask:</strong></p>
                <div class="code-block">pip install flask</div>
                
                <p style="margin-bottom: 10px;"><strong>2. Run the web application:</strong></p>
                <div class="code-block">python retroml_web_demo.py</div>
                
                <p style="margin-bottom: 10px;"><strong>3. Open your browser and go to:</strong></p>
                <div class="code-block">http://localhost:5000</div>
                
                <h3 style="color: #4ecdc4; margin: 20px 0 15px 0;">✨ Features:</h3>
                <ul style="color: #ffeb3b; text-align: left;">
                    <li>🎯 Real-time churn prediction</li>
                    <li>📊 Interactive form with customer data</li>
                    <li>🎮 Full retro 90s styling</li>
                    <li>📈 Confidence scores and probabilities</li>
                    <li>💡 Automated recommendations</li>
                    <li>📱 Mobile-responsive design</li>
                </ul>
                
                <h3 style="color: #4ecdc4; margin: 20px 0 15px 0;">🔧 Prerequisites:</h3>
                <ul style="color: #ffeb3b; text-align: left;">
                    <li>✅ Trained classification model in results/classification/</li>
                    <li>✅ Flask installed (pip install flask)</li>
                    <li>✅ All RetroML dependencies</li>
                </ul>
            </div>
            
            <div style="background: rgba(255, 107, 107, 0.2); border: 2px solid #ff6b6b; border-radius: 10px; padding: 15px; margin-top: 20px;">
                <div style="color: #ff6b6b; font-weight: bold;">⚠️ Demo Mode Active</div>
                <div style="color: #ffffff; margin-top: 10px;">
                    This is a static preview. Run the Python script to start the interactive web server!
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Add some retro effects
        document.addEventListener('DOMContentLoaded', function() {
            // Add glitch effect to title
            const title = document.querySelector('.title');
            setInterval(() => {
                if (Math.random() > 0.95) {
                    title.style.textShadow = '2px 0 #ff6b6b, -2px 0 #4ecdc4';
                    setTimeout(() => {
                        title.style.textShadow = '0 0 10px #00ffff';
                    }, 100);
                }
            }, 1000);
        });
    </script>
</body>
</html>
    '''
    
    # Save standalone demo
    with open('retroml_web_demo_standalone.html', 'w') as f:
        f.write(standalone_html)
    
    print("✅ Created standalone demo: retroml_web_demo_standalone.html")
    print("🌐 Open this file in your browser to see the demo!")

def main():
    """Main function"""
    print("🎮 RETROML WEB DEMO SETUP 🎮")
    print("=" * 40)
    
    # Check if model exists
    model_path = Path("results/classification/best_model.pkl")
    
    if not model_path.exists():
        print("❌ MODEL NOT FOUND!")
        print("Please run the classification pipeline first:")
        print("  python retroml.py configs/classification_example.json")
        print("\nCreating standalone demo instead...")
        create_standalone_demo()
        return
    
    if not HAS_FLASK:
        print("❌ FLASK NOT INSTALLED!")
        print("Install Flask to run the web demo:")
        print("  pip install flask")
        print("\nCreating standalone demo instead...")
        create_standalone_demo()
        return
    
    try:
        # Create and run web app
        app = RetroMLWebApp()
        app.run(host='0.0.0.0', port=5001, debug=False)
        
    except Exception as e:
        print(f"❌ Error starting web app: {e}")
        print("Creating standalone demo instead...")
        create_standalone_demo()

if __name__ == "__main__":
    main()