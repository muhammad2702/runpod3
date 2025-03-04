# Crypto Forecasting with LiteFormer

This repository contains a full pipeline for collecting, preprocessing, training, and forecasting cryptocurrency data using a transformer-based model called **LiteFormer**. The project integrates with the Polygon API to fetch historical crypto data, applies technical analysis indicators using the `ta` library, and builds a deep learning model with PyTorch to predict future close prices.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

The project implements a complete workflow for cryptocurrency forecasting:

1. **Data Collection:**  
   Fetches data from the Polygon API for given tickers and timeframes, saving raw data into CSV files.

2. **Data Preprocessing:**  
   Computes technical indicators such as RSI, MACD, ATR, Bollinger Bands, and ADX, and prepares the data for modeling.

3. **Model Training:**  
   Uses a transformer-based model (LiteFormer) with positional encoding for time series forecasting. Training includes features like gradient accumulation, learning rate scheduling, and early stopping.

4. **Prediction & Evaluation:**  
   Generates predictions for the next time steps, computes error metrics (MAE, RMSE), and categorizes predictions.

5. **Result Sharing:**  
   Optionally uploads the combined predictions CSV file to [Oshi](https://oshi.at) for easy sharing.

---

## Features

- **Multi-Ticker Data Collection:** Fetch historical data for multiple cryptocurrencies.
- **Technical Indicator Calculation:** Leverages the `ta` library for RSI, MACD, ATR, Bollinger Bands, and ADX.
- **Transformer-Based Forecasting:** Implements a custom transformer (LiteFormer) model using PyTorch.
- **Custom Dataset & DataLoader:** Prepares time-series windows from the data for model training.
- **Training & Validation Pipeline:** Uses gradient accumulation and adaptive learning rate scheduling.
- **Prediction & Metrics:** Evaluates model performance and saves predictions with additional classification (direction, percentage change).
- **File Upload:** Supports uploading results to Oshi for public access.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/crypto-forecasting.git
   cd crypto-forecasting
