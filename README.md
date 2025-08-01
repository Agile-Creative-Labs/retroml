# RetroML - 90s-Style Automated Machine Learning

Welcome to RetroML! A nostalgic, configuration-driven ML pipeline that brings the radical vibes of the 90s to modern machine learning.

## Quick Start

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
## Run the Demo for RetroML Classification Demo

1. **Run the interactive demo**
  ```bash
    python retroml_demo.py
  ```
2. **Analyze the model in detail**
  ```bash
    python3 retroml_model_analyzer.py
  ```    
3. **Ensure that you flask installed**
    ```bash
      pip install flask
    ```    
4. **Launch the web application**
  ```bash
    python3 retroml_web_demo.py
  ```

# Then open http://localhost:5000

## Project Structure

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

## Configuration Format

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

## Features

- ✅ **Automated model selection** - Tests multiple algorithms automatically
- ✅ **Config-driven workflow** - No coding required, just JSON configuration
- ✅ **Retro 90s UI** - ASCII art, colored output, loading animations
- ✅ **Comprehensive reporting** - Detailed evaluation reports and metrics
- ✅ **Model persistence** - Save and load trained models
- ✅ **Extensible architecture** - Easy to add new algorithms and preprocessing steps

## Supported Algorithms

**Classification:**
- Logistic Regression
- Random Forest
- XGBoost (if installed)

**Regression:**
- Linear Regression  
- Random Forest
- XGBoost (if installed)

## The Retro Experience

RetroML brings back the radical vibes of 90s computing:
- ASCII art banners and loading screens
- Colorful terminal output with Rich formatting
- Progress bars that look like old-school installers
- Nostalgic terminology and messaging

## Example Usage

```bash
# Run classification pipeline
python retroml.py configs/classification_example.json

# Run regression pipeline  
python retroml.py configs/regression_example.json

# Create your own config and run
python retroml.py configs/my_config.json
```

## Output Files

After running a pipeline, you'll find:
- `best_model.pkl` - The trained model (with preprocessing)
- `evaluation_report.txt` - Detailed performance metrics
- Console output with colorful retro styling

## Extending RetroML

Want to add new algorithms or preprocessing steps? The modular design makes it easy:

1. Add new models to `_get_default_models()`
2. Extend preprocessing options in `preprocess_data()`
3. Add new problem types (clustering, NLP, etc.)

## Coming Soon (Roadmap)

- Clustering algorithms (K-means, DBSCAN)
- NLP support with transformers
- Hyperparameter tuning with Optuna
- Model explainability with SHAP
- MLflow experiment tracking
- Web-based UI with Flask/Streamlit

---

*"Dude, this ML pipeline is totally tubular!" - RetroML, probably*

## Troubleshooting

**ModuleNotFoundError**: Install requirements with `pip install -r requirements.txt`

**FileNotFoundError**: Make sure your dataset path in the config is correct

**Performance Issues**: Try reducing dataset size or using simpler models for large datasets

## License

This project is open source. Feel free to modify and distribute!

---

🎮 **RetroML** - Bringing 90s vibes to modern ML! 🎮
