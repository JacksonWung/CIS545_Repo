# 📈 Multi-Modal Stock Price Prediction and Trading Strategy

A hybrid deep learning system for 5-minute AAPL stock price prediction, integrating technical indicators, fundamental analysis, and market sentiment data.

## 🎯 Project Overview

This project builds a hybrid deep learning model that predicts 5-minute closing prices for AAPL stock by integrating three data modalities:
- **Technical Analysis (TA)**: RSI, MACD, Bollinger Bands, and other time-series indicators
- **Fundamental Analysis (FA)**: P/E ratio, EPS, and other quarterly data
- **Sentiment Analysis (SA)**: FinBERT-based financial news sentiment scoring

## 🏗 Core Architecture

**Input Features** (8 core features total):
- LSTM Sequential Input (5-dim): Lagged close price, volume, RSI, MACD line, Bollinger Band width
- MLP Tabular Input (3-dim): P/E ratio, EPS, aggregated sentiment score

## 📁 Project Structure
cis545_quant/
<<<<<<< HEAD
├── data/ # Data storage
│ ├── raw/ # Raw datasets
│ └── processed/ # Processed features
├── src/
│ ├── data_loader.py # Data loading and preprocessing
│ ├── feature_engineer.py # Feature engineering pipeline
│ ├── config_loader.py # config loading
│ ├── model.py # Model architecture definitions
│ ├── train.py # Training procedures
│ └── backtest.py # Strategy backtesting
├── notebooks/ # Jupyter notebooks
│ ├── 01_EDA.ipynb # Exploratory Data Analysis
│ ├── 02_Feature_Engineering.ipynb
│ └── 03_Model_Training.ipynb
├── config/ # Configuration files
│ └── parameters.yaml # Model and data parameters
├── tests/ # Unit tests
├── requirements.txt # Python dependencies
└── README.md # Project documentation
=======
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

├── data/ # Data storage  
│ ├── raw/ # Raw datasets  
│ └── processed/ # Processed features  
├── src/  
│ ├── data_loader.py # Data loading and preprocessing  
│ ├── feature_engineer.py # Feature engineering pipeline  
│ ├── model.py # Model architecture definitions  
│ ├── train.py # Training procedures  
│ └── backtest.py # Strategy backtesting  
├── notebooks/ # Jupyter notebooks  
│ ├── 01_EDA.ipynb # Exploratory Data Analysis  
│ ├── 02_Feature_Engineering.ipynb  
│ └── 03_Model_Training.ipynb  
├── config/ # Configuration files  
│ └── parameters.yaml # Model and data parameters  
├── tests/ # Unit tests  
└── README.md # Project documentation  
>>>>>>> 28723f9c96c8d886837009c1829827e6024903a9


## 🗓 Project Timeline & Roadmap

### Phase 1: Data Preparation & EDA
**Goals**: Complete data collection, cleaning, and exploratory analysis
- [ ] Set up development environment and GitHub collaboration workflow
- [ ] Collect AAPL 5-minute OHLCV data (yfinance/Kaggle)
- [ ] Acquire fundamental data (P/E, EPS) and financial news datasets
- [ ] Execute key EDA visualizations:
  - High-frequency price/volume charts
  - 5-minute return distribution histograms
  - Technical indicator overlay analysis
  - Feature correlation heatmaps

### Phase 2: Feature Engineering
**Goals**: Construct complete 8-dimensional feature set
- [ ] Calculate technical indicators: RSI(14), MACD(12,26,9), Bollinger Band width
- [ ] Implement FinBERT sentiment analysis pipeline
- [ ] Design weighted sentiment score aggregation strategy
- [ ] Time alignment for fundamental data (forward filling)
- [ ] Data normalization and stationarity processing

### Phase 3: Model Development 
**Goals**: Build and train hybrid deep learning model
- [ ] Implement Keras Functional API dual-input architecture
- [ ] Establish LSTM baseline model for performance benchmarking
- [ ] Develop hybrid LSTM-MLP model
- [ ] Hyperparameter tuning and regularization (Dropout)
- [ ] Time-series cross-validation training

### Phase 4: Model Evaluation 
**Goals**: Comprehensive model performance and feature contribution analysis
- [ ] Calculate regression metrics: RMSE, MAE, R²
- [ ] Execute SHAP analysis for feature importance quantification
- [ ] Compare hybrid model vs. baseline model performance
- [ ] Analyze multi-modal feature synergy effects

### Phase 5: Trading Strategy & Backtesting 
**Goals**: Transform predictive model into executable trading strategy
- [ ] Design trading signals based on predicted returns
- [ ] Implement risk management rules (stop-loss/take-profit)
- [ ] Build backtesting framework (Backtrader/Zipline)
- [ ] Incorporate transaction costs (commissions, slippage)
- [ ] Calculate risk-adjusted performance metrics

### Phase 6: Final Optimization & Documentation
**Goals**: Project completion and results presentation
- [ ] Final model performance optimization
- [ ] Generate comprehensive project documentation
- [ ] Prepare presentation materials and final report
- [ ] Code cleanup and repository maintenance
