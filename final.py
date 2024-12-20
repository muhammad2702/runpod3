import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from ta import trend, momentum, volatility
import pickle
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_uniform_ as xavier_uniform
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import logging
from itertools import product
import json  # For JSON serialization
import sys   # To read input from stdin
import runpod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Metrics Dictionary
metrics_dict = {
    "status": "pending",
    "message": "",
    "details": {}
}

# Configuration - Securely load API keys using environment variables
API_KEY = 'de_kgSuhw6v4KnRK0wprJCoBAIhqSd5R'  # Replace with your actual API key
BASE_URL = 'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'

# List of cryptocurrencies (tickers) you want to collect data for
TICKERS = [
    'X:AAVEUSD',
    'X:AVAXUSD',
    'X:BATUSD',
    'X:LINKUSD',
    'X:UNIUSD',
    'X:SUSHIUSD',
    'X:PNGUSD',
    'X:JOEUSD',
    'X:XAVAUSD',
    'X:ATOMUSD',
    'X:ALGOUSD',
    'X:ARBUSD',
    'X:1INCHUSD',
    'X:DAIUSD',
    # Add more tickers as needed
]

# Timeframes you want to collect data for
TIMEFRAMES = [
    {'multiplier': 1, 'timespan': 'second'},
]

DATA_DIR = 'crypto_data'
os.makedirs(DATA_DIR, exist_ok=True)

MODELS_DIR = 'models'
SCALERS_DIR = 'scalers'
PREDICTIONS_DIR = 'predictions'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)



def fetch_data(ticker, multiplier, timespan, from_date, to_date):
    url = BASE_URL.format(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date.strftime('%Y-%m-%d'),
        to_date=to_date.strftime('%Y-%m-%d')
    )
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': '50000',
        'apiKey': API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred for {ticker} - {timespan}: {http_err}")
    except Exception as err:
        logging.error(f"Other error occurred for {ticker} - {timespan}: {err}")
    return []

def collect(START_DATE, END_DATE):
    start = datetime.strptime(START_DATE, '%Y-%m-%d')
    end = datetime.strptime(END_DATE, '%Y-%m-%d')

    for ticker in TICKERS:
        ticker_dir = os.path.join(DATA_DIR, ticker.replace(":", "_"))
        os.makedirs(ticker_dir, exist_ok=True)

        for timeframe in TIMEFRAMES:
            multiplier = timeframe['multiplier']
            timespan = timeframe['timespan']
            filename = f"{ticker.replace(':', '_')}_{multiplier}{timespan}.csv"
            filepath = os.path.join(ticker_dir, filename)

            logging.info(f"Fetching data for {ticker} - {multiplier}{timespan}")
            data = fetch_data(ticker, multiplier, timespan, start, end)

            if data:
                df = pd.DataFrame(data)
                df['t'] = pd.to_datetime(df['t'], unit='ms')
                df.to_csv(filepath, index=False)
                logging.info(f"Saved {len(df)} records to {filepath}")
            else:
                logging.warning(f"No data fetched for {ticker} - {multiplier}{timespan}")

            time.sleep(1)

class CryptoDataPreprocessor:
    def __init__(self, raw_data_dir='crypto_data', preprocessed_data_dir='preprocessed_data', columns_to_add=None):
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir
        self.columns_to_add = columns_to_add or ['close_price']
        os.makedirs(self.preprocessed_data_dir, exist_ok=True)

    def preprocess_file(self, df):
        required_columns = ['c', 'h', 'l']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        rsi = momentum.RSIIndicator(close=df['c'], window=14)
        df['RSI'] = rsi.rsi()

        macd = trend.MACD(close=df['c'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd().rolling(window=3).mean()
        df['MACD_signal'] = macd.macd_signal().rolling(window=3).mean()
        df['MACD_diff'] = macd.macd_diff().rolling(window=3).mean()

        atr = volatility.AverageTrueRange(high=df['h'], low=df['l'], close=df['c'], window=14)
        df['ATR'] = atr.average_true_range().rolling(window=3).mean()

        bollinger = volatility.BollingerBands(close=df['c'], window=20, window_dev=2)
        df['BB_upper'] = bollinger.bollinger_hband().rolling(window=3).mean()
        df['BB_lower'] = bollinger.bollinger_lband().rolling(window=3).mean()
        df['BB_width'] = ((df['BB_upper'] - df['BB_lower']) / df['c']).rolling(window=3).mean()

        adx = trend.ADXIndicator(high=df['h'], low=df['l'], close=df['c'], window=14)
        df['ADX'] = adx.adx().rolling(window=3).mean()

        df.dropna(inplace=True)
        df['close_price'] = df['c']

        return df

    def save_preprocessed_data(self, df, filepath):
        df[self.columns_to_add].to_csv(filepath, index=False)
        logging.info(f"Saved preprocessed data to {filepath}")

    def preprocess_all_files(self):
        for ticker in os.listdir(self.raw_data_dir):
            ticker_raw_dir = os.path.join(self.raw_data_dir, ticker)
            ticker_preprocessed_dir = os.path.join(self.preprocessed_data_dir, ticker)
            
            if os.path.exists(ticker_preprocessed_dir):
                for old_file in os.listdir(ticker_preprocessed_dir):
                    old_file_path = os.path.join(ticker_preprocessed_dir, old_file)
                    try:
                        os.remove(old_file_path)
                        logging.debug(f"Removed old preprocessed file: {old_file_path}")
                    except Exception as e:
                        logging.error(f"Error removing file {old_file_path}: {e}")
            else:
                os.makedirs(ticker_preprocessed_dir, exist_ok=True)

            for file in os.listdir(ticker_raw_dir):
                if file.endswith('.csv'):
                    raw_filepath = os.path.join(ticker_raw_dir, file)
                    preprocessed_filename = file.replace('.csv', '_preprocessed.csv')
                    preprocessed_filepath = os.path.join(ticker_preprocessed_dir, preprocessed_filename)

                    try:
                        df_raw = pd.read_csv(raw_filepath)
                        if 't' in df_raw.columns and not pd.api.types.is_datetime64_any_dtype(df_raw['t']):
                            df_raw['t'] = pd.to_datetime(df_raw['t'], errors='coerce')
                    except Exception as e:
                        logging.error(f"Error reading {raw_filepath}: {e}")
                        continue

                    df_raw.dropna(subset=['t'], inplace=True)

                    try:
                        df_preprocessed = self.preprocess_file(df_raw)
                    except Exception as e:
                        logging.error(f"Error preprocessing {raw_filepath}: {e}")
                        continue

                    logging.info(f"Preprocessed data for {file}:")
                    print(df_preprocessed.head())

                    self.save_preprocessed_data(df_preprocessed, preprocessed_filepath)
                    logging.info(f"Saved preprocessed file to {preprocessed_filepath}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length, :]
        return x

class LiteFormer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1, max_seq_length=80):
        super(LiteFormer, self).__init__()
        
        self.d_model = d_model
        self.input_linear = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.output_layer = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, src):
        src = self.input_linear(src)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        
        memory = self.transformer_encoder(src)
        memory = torch.mean(memory, dim=1)
        
        out = self.output_layer(memory)
        return out

class CryptoDataset(Dataset):
    def __init__(self, dataframe, window_size=80):
        self.data = dataframe.sort_values('t').reset_index(drop=True)
        self.window_size = window_size
        self.length = len(self.data) - self.window_size - 1 if len(self.data) > self.window_size else 0

    def __len__(self):
        return max(self.length, 0)

    def __getitem__(self, idx):
        data = self.data.iloc[idx:idx+self.window_size]
        features = data['close_price'].values.reshape(-1, 1)
        target_idx = idx + self.window_size
        target = self.data.iloc[target_idx]['close_price']
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def train(model, dataloader, criterion, optimizer, scheduler, scaler, device, accumulation_steps=2):
    model.train()
    epoch_loss = 0.0
    total = 0
    optimizer.zero_grad(set_to_none=True)

    for idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accumulation_steps

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * accumulation_steps * inputs.size(0)
        total += inputs.size(0)

        if (idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # Final step if not divisible by accumulation steps
    if (idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    epoch_loss /= total if total > 0 else 1
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    val_loss /= total if total > 0 else 1
    mae = mean_absolute_error(all_targets, all_preds) if len(all_targets)>0 else 0
    rmse = math.sqrt(mean_squared_error(all_targets, all_preds)) if len(all_targets)>0 else 0
    return val_loss, mae, rmse

def plot_losses(train_losses, val_losses, save_path='loss_curve.png'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Loss curve saved to {save_path}")

def main(crypto_metrics):
    # Train a separate model for each crypto individually
    batch_size = 48
    epochs = 2
    learning_rate = 5e-4
    window_size = 80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessed_data_dir = 'preprocessed_data'
    cryptos = [d for d in os.listdir(preprocessed_data_dir) if os.path.isdir(os.path.join(preprocessed_data_dir, d))]

    for crypto in cryptos:
        crypto_metrics[crypto] = {
            "training_losses": [],
            "validation_losses": [],
            "test_metrics": {},
            "prediction_metrics": {}  # Added to store prediction metrics
        }

        # Load preprocessed CSV files for this crypto
        ticker_dir = os.path.join(preprocessed_data_dir, crypto)
        csv_files = [f for f in os.listdir(ticker_dir) if f.endswith('_preprocessed.csv')]
        if not csv_files:
            logging.warning(f"No preprocessed data found for {crypto}. Skipping.")
            continue

        # Concatenate all files for this crypto
        df_list = []
        for f in csv_files:
            fp = os.path.join(ticker_dir, f)
            df = pd.read_csv(fp)
            df['crypto'] = crypto
            df_list.append(df)
        full_df = pd.concat(df_list, ignore_index=True)
        full_df.dropna(subset=['close_price'], inplace=True)

        if len(full_df) <= window_size:
            logging.warning(f"Not enough data for {crypto} to train. Skipping.")
            continue

        # Split data
        train_size = int(0.7 * len(full_df))
        val_size = int(0.15 * len(full_df))
        train_df = full_df.iloc[:train_size].copy()
        val_df = full_df.iloc[train_size:train_size+val_size].copy()
        test_df = full_df.iloc[train_size+val_size:].copy()

        # Scale individually for this crypto
        scaler = StandardScaler()
        scaler.fit(train_df[['close_price']])
        train_df['close_price'] = scaler.transform(train_df[['close_price']])
        val_df['close_price'] = scaler.transform(val_df[['close_price']])
        test_df['close_price'] = scaler.transform(test_df[['close_price']])
        
        # Save scaler
        scaler_path = os.path.join(SCALERS_DIR, f'{crypto}_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler for {crypto} saved to {scaler_path}")

        # Create datasets
        train_dataset = CryptoDataset(train_df, window_size)
        val_dataset = CryptoDataset(val_df, window_size)
        test_dataset = CryptoDataset(test_df, window_size)

        if len(train_dataset) == 0:
            logging.warning(f"No training samples for {crypto} after windowing. Skipping.")
            continue

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Initialize model
        model = LiteFormer(
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            max_seq_length=window_size
        ).to(device)
        logging.info(f"LiteFormer model initialized for {crypto}.")

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        scaler_amp = torch.cuda.amp.GradScaler()

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        early_stopping_patience = 15

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs} for {crypto}")
            train_loss = train(model, train_loader, criterion, optimizer, scheduler, scaler_amp, device, accumulation_steps=2)
            train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.4f}")
            crypto_metrics[crypto]["training_losses"].append(train_loss)

            val_loss, val_mae, val_rmse = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
            crypto_metrics[crypto]["validation_losses"].append({
                "val_loss": val_loss,
                "mae": val_mae,
                "rmse": val_rmse
            })

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                logging.info(f"New best model for {crypto} saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter > early_stopping_patience:
                    logging.info("Early stopping triggered.")
                    break

            gc.collect()

        # Plot losses if you like
        plot_path = f'loss_curve_{crypto}.png'
        plot_losses(train_losses, val_losses, save_path=plot_path)

        # Save the best model
        if best_model_state is not None:
            model_path = os.path.join(MODELS_DIR, f'best_liteformer_model_{crypto}.pth')
            torch.save(best_model_state, model_path)
            logging.info(f"Best model for {crypto} saved as {model_path}")
            model.load_state_dict(best_model_state)
        else:
            logging.warning(f"No improvement during training for {crypto}, model not saved.")

        # Final evaluation
        test_loss, test_mae, test_rmse = validate(model, test_loader, criterion, device)
        print(f"\nFinal Test Results for {crypto}:")
        print(f"Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
        crypto_metrics[crypto]["test_metrics"] = {
            "test_loss": test_loss,
            "mae": test_mae,
            "rmse": test_rmse
        }

        del model
        torch.cuda.empty_cache()

def preprocess_and_predict(crypto_metrics):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessed_data_dir = 'preprocessed_data'
    cryptos = [d for d in os.listdir(preprocessed_data_dir) if os.path.isdir(os.path.join(preprocessed_data_dir, d))]

    all_predictions = []
    for crypto in cryptos:
        ticker_dir = os.path.join(preprocessed_data_dir, crypto)
        csv_files = [f for f in os.listdir(ticker_dir) if f.endswith('_preprocessed.csv')]
        if not csv_files:
            logging.warning(f"No preprocessed data found for prediction: {crypto}. Skipping.")
            continue

        df_list = []
        for f in csv_files:
            fp = os.path.join(ticker_dir, f)
            try:
                df = pd.read_csv(fp)
                df['crypto'] = crypto
                df_list.append(df)
            except Exception as e:
                logging.error(f"Error reading {fp}: {e}")
                continue
        if not df_list:
            continue

        full_df = pd.concat(df_list, ignore_index=True)
        full_df.dropna(subset=['close_price'], inplace=True)
        if len(full_df) <= 80:
            logging.warning(f"Not enough data for prediction: {crypto}. Skipping.")
            continue

        # Load scaler for this crypto
        scaler_path = os.path.join(SCALERS_DIR, f'{crypto}_scaler.joblib')
        if not os.path.exists(scaler_path):
            logging.warning(f"No scaler found for {crypto}. Skipping prediction.")
            continue
        scaler = joblib.load(scaler_path)
        full_df['close_price'] = scaler.transform(full_df[['close_price']])

        # Dataset and DataLoader
        test_dataset = CryptoDataset(full_df, window_size=80)
        if len(test_dataset) == 0:
            logging.warning(f"No samples for prediction: {crypto}. Skipping.")
            continue
        test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=2)

        # Load model
        model_path = os.path.join(MODELS_DIR, f'best_liteformer_model_{crypto}.pth')
        if not os.path.exists(model_path):
            logging.warning(f"No model found for {crypto}. Skipping prediction.")
            continue

        model = LiteFormer(
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            max_seq_length=80
        ).to(device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        except Exception as e:
            logging.error(f"Error loading model for {crypto}: {e}")
            continue

        predictions = []
        prediction_timestamps = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs = outputs.cpu().numpy().flatten()
                predictions.extend(outputs.tolist())

                start_index = len(predictions) - len(outputs)
                end_index = start_index + len(outputs)
                # Align timestamps
                corresponding_timestamps = full_df['t'].iloc[start_index + 80:end_index + 80].tolist()
                prediction_timestamps.extend(corresponding_timestamps)

        # Create predictions_df
        predictions_df = pd.DataFrame({
            't': prediction_timestamps,
            'crypto': full_df['crypto'].iloc[80:80 + len(predictions)].tolist(),
            'predicted_close_price': predictions
        })

        # Inverse transform predictions
        predictions_df['predicted_close_price'] = scaler.inverse_transform(predictions_df[['predicted_close_price']])

        # Retrieve true close prices for comparison
        true_close_prices = full_df['close_price'].iloc[80:80 + len(predictions)]
        true_close_prices = scaler.inverse_transform(true_close_prices.values.reshape(-1, 1)).flatten()

        # Compute metrics
        if len(true_close_prices) > 0:
            mae = mean_absolute_error(true_close_prices, predictions_df['predicted_close_price'])
            rmse = math.sqrt(mean_squared_error(true_close_prices, predictions_df['predicted_close_price']))
        else:
            mae = 0
            rmse = 0

        print(f"Prediction MAE for {crypto}: {mae:.4f}, RMSE: {rmse:.4f}")

        # Store prediction metrics
        if crypto in crypto_metrics:
            crypto_metrics[crypto]["prediction_metrics"] = {
                "mae": mae,
                "rmse": rmse
            }
        else:
            crypto_metrics[crypto] = {
                "prediction_metrics": {
                    "mae": mae,
                    "rmse": rmse
                }
            }

        # Add classification columns
        last_actual_close = true_close_prices
        predictions_df['direction'] = np.where(predictions_df['predicted_close_price'] > last_actual_close, 'Up', 'Down')
        predictions_df['percentage_change'] = ((predictions_df['predicted_close_price'] - last_actual_close) / last_actual_close) * 100
        predictions_df['last_actual_close'] = last_actual_close

        def categorize_change(pc):
            if pc >= 2:
                return 'Significant Up'
            elif 0.5 <= pc < 2:
                return 'Moderate Up'
            elif -0.5 < pc < 0.5:
                return 'No Change'
            elif -2 < pc <= -0.5:
                return 'Moderate Down'
            else:
                return 'Significant Down'

        predictions_df['change_category'] = predictions_df['percentage_change'].apply(categorize_change)
        predictions_df['percentage_change'] = predictions_df['percentage_change'].round(2)

        # Save per-crypto predictions
        crypto_prediction_path = os.path.join(PREDICTIONS_DIR, f'{crypto}_latest_predictions.csv')
        predictions_df.to_csv(crypto_prediction_path, index=False)
        logging.info(f"Predictions for {crypto} saved to {crypto_prediction_path}")

        all_predictions.append(predictions_df)

    # Merge all predictions into one file if desired
    if all_predictions:
        final_predictions_df = pd.concat(all_predictions, ignore_index=True)
        final_path = os.path.join(PREDICTIONS_DIR, 'all_latest_predictions.csv')
        final_predictions_df.to_csv(final_path, index=False)
        logging.info(f"All predictions combined saved to {final_path}")
        print(final_predictions_df.head())

    return {"status": "success", "message": "Predictions completed."}




def upload_to_oshi(file_path):
    try:
        with open(file_path, 'rb') as file:
            files = {'f': file}
            response = requests.post('https://oshi.at', files=files)
            response.raise_for_status()
            # Extract the download link from the response text
            download_link = response.text.strip()
            return download_link
    except Exception as e:
        print(f"An error occurred during file upload: {e}")
        return None



def handler(job):
    job_input = job.get("input", {})
    
    # Retrieve the 'START_DATE' and 'END_DATE' values
    START_DATE1 = job_input.get("START_DATE1", "")
    print(f"START_DATE1 :  {START_DATE1}")
    END_DATE1 = job_input.get("END_DATE1", "")
    print(f"END_DATE1 :  {END_DATE1}")

    collect(START_DATE1, END_DATE1)
    # Uncomment if you want to run these steps
    preprocess = CryptoDataPreprocessor(
         raw_data_dir='crypto_data',
         preprocessed_data_dir='preprocessed_data',
         columns_to_add=['close_price', 't']
     )
    preprocess.preprocess_all_files()

    crypto_metrics = {}
    main(crypto_metrics)

    START_DATE2 = job_input.get("START_DATE2", "")
    print(f"START_DATE2 :  {START_DATE2}")
    END_DATE2 = job_input.get("END_DATE2", "")
    print(f"END_DATE2 :  {END_DATE2}")

    collect(START_DATE2, END_DATE2)
    preprocess = CryptoDataPreprocessor(
        raw_data_dir='crypto_data',
        preprocessed_data_dir='preprocessed_data',
        columns_to_add=['close_price', 't']
    )
    preprocess.preprocess_all_files()

    # Step 2: Predictions
    prediction_status = preprocess_and_predict(crypto_metrics)  # Pass crypto_metrics to collect prediction metrics
    if prediction_status["status"] != "success":
        logging.error("Prediction step failed. Exiting.")
        metrics_dict["status"] = "failed"
        metrics_dict["message"] = "Prediction step failed."
        print(json.dumps(metrics_dict))
        return json.dumps(metrics_dict)  # Ensure handler exits after failure

    # Step 3: Load the combined predictions
    all_predictions_path = os.path.join(PREDICTIONS_DIR, 'all_latest_predictions.csv')
    if not os.path.exists(all_predictions_path):
        logging.error(f"Combined predictions file not found at {all_predictions_path}. Exiting.")
        metrics_dict["status"] = "failed"
        metrics_dict["message"] = f"Combined predictions file not found at {all_predictions_path}."
        print(json.dumps(metrics_dict))
        return json.dumps(metrics_dict)  # Ensure handler exits after failure

    predictions_df = pd.read_csv(all_predictions_path)
    print(predictions_df)
    download_link = upload_to_oshi(all_predictions_path)
    print(download_link)
   
        
    metrics_dict["download_link"] = download_link

    
    metrics_dict["status"] = "success"
    metrics_dict["message"] = "Processing completed successfully."
    metrics_dict["details"] = crypto_metrics
    
    return json.dumps(metrics_dict)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# =========================
# Main Execution
# =========================

runpod.serverless.start({"handler": handler})
