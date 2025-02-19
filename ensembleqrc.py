import numpy as np
import pandas as pd
import yfinance as yf
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class QuantumReservoirComputer:
    def __init__(self, n_qubits=6, n_reservoir_neurons=32, connectivity=0.7, seed=42):
        self.n_qubits = n_qubits
        self.n_reservoir_neurons = n_reservoir_neurons
        self.connectivity = connectivity
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        np.random.seed(seed)
        self.reservoir_weights = np.random.normal(
            0, 1/np.sqrt(n_qubits), 
            size=(n_reservoir_neurons, n_qubits)
        ) * (np.random.rand(n_reservoir_neurons, n_qubits) < connectivity)

    def quantum_reservoir_layer(self, inputs):
        @qml.qnode(self.dev)
        def quantum_circuit(x):
            # Enhanced amplitude encoding
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i] * np.pi, wires=i)
                qml.RZ(x[i] * np.pi / 2, wires=i)
            
            # Multi-layer entanglement structure
            for _ in range(3):
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(np.pi / 4, wires=j)
                        qml.RY(np.pi / 6, wires=i)
                
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RY(np.pi / 2, wires=i)
                    qml.RZ(np.pi / 4, wires=i)
            
            measurements = []
            measurements.extend([qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)])
            measurements.extend([qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)])
            measurements.extend([qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)])
            return measurements
        
        return quantum_circuit(inputs)

    def generate_reservoir_states(self, X):
        reservoir_states = []
        for sample in X:
            projected_sample = np.tanh(np.dot(self.reservoir_weights, sample[:self.n_qubits]))
            quantum_state = self.quantum_reservoir_layer(projected_sample)
            reservoir_states.append(quantum_state)
        return np.array(reservoir_states)

class StockTrendPredictor:
    def __init__(self, ticker="AAPL", start_date="2020-01-01", end_date="2024-01-16"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = StandardScaler()

    def _calculate_rsi(self, prices, periods=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        return exp1 - exp2

    def _calculate_atr(self, data, period=14):
        high = data['High']
        low = data['Low']
        close = data['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_bollinger_bands(self, data, window=20):
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band - lower_band

    def fetch_and_preprocess_data(self, lookback_window=30):
        print(f"Fetching stock data for {self.ticker} from {self.start_date} to {self.end_date}...")
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        if stock_data.empty:
            raise ValueError("No stock data available. Please check the ticker or date range.")

        # Calculate daily returns first
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        
        # Multiple timeframe analysis
        for window in [5, 10, 20, 50]:
            stock_data[f'SMA_{window}'] = stock_data['Close'].rolling(window=window).mean()
            stock_data[f'EMA_{window}'] = stock_data['Close'].ewm(span=window).mean()
        
        # RSI with multiple periods
        for period in [7, 14, 21]:
            stock_data[f'RSI_{period}'] = self._calculate_rsi(stock_data['Close'], periods=period)
        
        # MACD variations
        stock_data['MACD'] = self._calculate_macd(stock_data['Close'])
        stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9).mean()
        stock_data['MACD_Hist'] = stock_data['MACD'] - stock_data['MACD_Signal']
        
        # Additional technical indicators
        stock_data['ATR'] = self._calculate_atr(stock_data[['High', 'Low', 'Close']], period=14)
        stock_data['BB_Width'] = self._calculate_bollinger_bands(stock_data)
        
        # Return features
        for period in [1, 3, 5, 10]:
            stock_data[f'Return_{period}d'] = stock_data['Close'].pct_change(period)
            stock_data[f'Volume_Change_{period}d'] = stock_data['Volume'].pct_change(period)
        
        # Momentum indicators
        stock_data['Price_Acceleration'] = stock_data['Daily_Return'].diff()
        stock_data['Volume_Acceleration'] = stock_data['Volume'].pct_change().diff()
        
        # Volatility features
        for window in [5, 10, 20]:
            stock_data[f'Volatility_{window}d'] = stock_data['Daily_Return'].rolling(window=window).std()
        
        # Price patterns and levels
        stock_data['Higher_High'] = (stock_data['High'] > stock_data['High'].shift(1)).astype(int)
        stock_data['Lower_Low'] = (stock_data['Low'] < stock_data['Low'].shift(1)).astype(int)
        stock_data['Support'] = stock_data['Low'].rolling(window=lookback_window).min()
        stock_data['Resistance'] = stock_data['High'].rolling(window=lookback_window).max()
        stock_data['Price_to_Support'] = (stock_data['Close'] - stock_data['Support']) / stock_data['Support']
        stock_data['Price_to_Resistance'] = (stock_data['Resistance'] - stock_data['Close']) / stock_data['Close']
        
        # Create trend labels using multiple timeframe consensus
        short_term = stock_data['Daily_Return'].rolling(window=5).mean()
        medium_term = stock_data['Daily_Return'].rolling(window=10).mean()
        long_term = stock_data['Daily_Return'].rolling(window=20).mean()
        
        # Combine signals from multiple timeframes
        stock_data['Trend'] = ((short_term > 0).astype(int) + 
                              (medium_term > 0).astype(int) + 
                              (long_term > 0).astype(int))
        # Convert to binary (majority vote)
        stock_data['Trend'] = (stock_data['Trend'] >= 2).astype(int)
        
        # Select features
        features = [col for col in stock_data.columns 
                   if col not in ['Trend', 'Daily_Return'] and 
                   not col.startswith('Adj')]
        
        # Drop rows with NaN values
        stock_data = stock_data.dropna()
        
        X = stock_data[features].values
        y = stock_data['Trend'].values
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Data preprocessing complete. {len(stock_data)} records processed.")
        return X_scaled, y

    def predict_trend(self, n_qubits=6, n_reservoir_neurons=32, connectivity=0.7, n_ensemble_members=11):
        X, y = self.fetch_and_preprocess_data()
        
        # Use more recent data for testing
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        predictions = []
        ensemble_weights = []
        
        print(f"\nTraining ensemble of {n_ensemble_members} quantum reservoirs...")
        val_split = int(len(X_train) * 0.8)
        X_train_main, X_val = X_train[:val_split], X_train[val_split:]
        y_train_main, y_val = y_train[:val_split], y_train[val_split:]
        
        for i in range(n_ensemble_members):
            print(f"Training ensemble member {i+1}/{n_ensemble_members}")
            
            qrc = QuantumReservoirComputer(
                n_qubits=n_qubits,
                n_reservoir_neurons=n_reservoir_neurons,
                connectivity=connectivity,
                seed=42 + i * 17
            )
            
            X_train_reservoir = qrc.generate_reservoir_states(X_train_main)
            X_val_reservoir = qrc.generate_reservoir_states(X_val)
            X_test_reservoir = qrc.generate_reservoir_states(X_test)
            
            best_score = 0
            best_model = None
            for C in [0.01, 0.1, 0.5, 1.0]:
                readout = LogisticRegression(
                    max_iter=3000,
                    C=C,
                    class_weight='balanced',
                    random_state=42 + i
                )
                readout.fit(X_train_reservoir, y_train_main)
                val_score = readout.score(X_val_reservoir, y_val)
                
                if val_score > best_score:
                    best_score = val_score
                    best_model = readout
            
            pred_proba = best_model.predict_proba(X_test_reservoir)[:, 1]
            predictions.append(pred_proba)
            ensemble_weights.append(best_score)
        
        ensemble_weights = np.array(ensemble_weights) / sum(ensemble_weights)
        ensemble_pred = np.average(predictions, axis=0, weights=ensemble_weights)
        
        threshold = np.percentile(ensemble_pred, 50)
        y_pred = (ensemble_pred > threshold).astype(int)
        
        print(f"\nResults for {n_ensemble_members} ensemble members:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
        return y_pred, y_test

def test_ensemble_sizes(max_members=25, step=2):
    accuracies = []
    roc_aucs = []
    ensemble_sizes = range(1, max_members + 1, step)
    
    plt.figure(figsize=(12, 6))
    
    for n_ensemble in ensemble_sizes:
        predictor = StockTrendPredictor()
        
        try:
            y_pred, y_test = predictor.predict_trend(
                n_qubits=6,
                n_reservoir_neurons=32,
                connectivity=0.7,
                n_ensemble_members=n_ensemble
            )
            
            acc = accuracy_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            
            accuracies.append(acc)
            roc_aucs.append(roc)
            
        except Exception as e:
            print(f"Error with {n_ensemble} members: {e}")
            accuracies.append(None)
            roc_aucs.append(None)
    
    # Plot results
    plt.plot(list(ensemble_sizes), accuracies, 'b-', label='Accuracy')
    plt.plot(list(ensemble_sizes), roc_aucs, 'r-', label='ROC-AUC')
    plt.xlabel('Number of Ensemble Members')
    plt.ylabel('Score')
    plt.title('Performance vs Ensemble Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('ensemble_size_results.png')
    plt.close()
    
    # Find optimal size
    best_acc_idx = np.argmax(accuracies)
    best_roc_idx = np.argmax(roc_aucs)
    
    print("\nResults Summary:")
    print(f"Best Accuracy: {accuracies[best_acc_idx]:.4f} with {list(ensemble_sizes)[best_acc_idx]} members")
    print(f"Best ROC-AUC: {roc_aucs[best_roc_idx]:.4f} with {list(ensemble_sizes)[best_roc_idx]} members")
    
    return accuracies, roc_aucs, list(ensemble_sizes)

def main():
    # Test different ensemble sizes
    print("Testing different ensemble sizes...")
    accuracies, roc_aucs, sizes = test_ensemble_sizes(max_members=25, step=2)

if __name__ == "__main__":
    main()
