import numpy as np
import pandas as pd
import yfinance as yf

# Quantum libraries
import pennylane as qml

# Classical ML
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout

# Suppress unnecessary warnings (optional)
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


# ---------------------------------------------------------------------
# 1) DATA FETCHING & PREPROCESSING
# ---------------------------------------------------------------------
def fetch_and_preprocess_data(ticker="AAPL", start_date="2020-01-01", end_date="2024-01-01"):
    """
    Fetches OHLCV data from Yahoo Finance, then creates a basic feature set:
       - daily returns for Close, High, Low, Volume
       - a simple 5-day moving average of Close
    Defines a binary Trend label: 1 if next day's Close > today's Close, else 0.
    """
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        raise ValueError("No data returned. Check ticker/date range.")

    # Create features
    df["Close_Return"] = df["Close"].pct_change()
    df["High_Return"] = df["High"].pct_change()
    df["Low_Return"] = df["Low"].pct_change()
    df["Volume_Return"] = df["Volume"].pct_change()
    df["SMA_5"] = df["Close"].rolling(5).mean()

    # Forward fill the SMA to avoid too many NaNs
    df["SMA_5"].fillna(method="bfill", inplace=True)

    # Define label: 1 if tomorrow's Close > today's Close, else 0
    df["Trend"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop rows with NaNs (from pct_change, rolling, shift)
    df.dropna(inplace=True)

    # Prepare X, y
    features = ["Close_Return", "High_Return", "Low_Return", "Volume_Return", "SMA_5"]
    X = df[features].values
    y = df["Trend"].values

    print(f"Total records after preprocessing: {len(df)}")
    return X, y


# ---------------------------------------------------------------------
# 2) QUANTUM RESERVOIR COMPUTING CLASS
# ---------------------------------------------------------------------
class QuantumReservoirComputer:
    """
    Example QRC implementation where the number of reservoir neurons
    matches the number of qubits (and input features), to avoid dimension
    mismatches and fully utilize each dimension.
    """
    def __init__(self, n_qubits=5, n_reservoir_neurons=5, connectivity=0.7, seed=42):
        """
        - n_qubits, n_reservoir_neurons: set to 5 each, so shape is (5,5).
        - connectivity: fraction of non-zero weights.
        """
        self.n_qubits = n_qubits
        self.n_reservoir_neurons = n_reservoir_neurons
        self.connectivity = connectivity

        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Seed for reproducibility
        np.random.seed(seed)

        # Build reservoir weights: shape (5, 5)
        # We'll zero-out some entries based on connectivity
        base_weights = np.random.normal(
            0, 1 / np.sqrt(self.n_qubits),
            size=(self.n_reservoir_neurons, self.n_qubits)
        )
        mask = (np.random.rand(self.n_reservoir_neurons, self.n_qubits) < connectivity)
        self.reservoir_weights = base_weights * mask

    def quantum_layer(self, inputs):
        """
        A simple quantum circuit that:
          - Encodes 'inputs' (size = n_qubits) as RY rotations
          - Applies a full set of CNOT entangling gates
          - Returns expectation values of PauliZ on each qubit
        """
        @qml.qnode(self.dev)
        def circuit(x):
            # Encode each dimension
            for i in range(self.n_qubits):
                qml.RY(x[i] * np.pi, wires=i)

            # Simple entangling layer
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])

            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit(inputs)

    def generate_reservoir_states(self, X):
        """
        For each sample in X (shape (n_features=5,)):
          1) project onto reservoir_weights => shape (5,)
          2) apply nonlinearity (tanh)
          3) feed into quantum_layer => shape (5,)
        Returns array of shape (len(X), 5).
        """
        reservoir_states = []
        for sample in X:
            # Project sample -> shape (5,)
            projected_sample = np.dot(self.reservoir_weights, sample)
            projected_sample = np.tanh(projected_sample)

            # Quantum circuit output -> shape (5,)
            q_out = self.quantum_layer(projected_sample)
            reservoir_states.append(q_out)

        return np.array(reservoir_states)


# ---------------------------------------------------------------------
# 3) MAIN: TRAIN/COMPARE QRC & CLASSICAL MODELS
# ---------------------------------------------------------------------
def main():
    # 1. Fetch data
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    X, y = fetch_and_preprocess_data(ticker, start_date, end_date)

    # 2. Train/Test Split (chronologically)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------------------------------------------
    # 3.1 Quantum Reservoir Computing
    # ----------------------------------------------------------------
    print("\n=== Quantum Reservoir Computing (QRC) ===")
    qrc = QuantumReservoirComputer(
        n_qubits=5,
        n_reservoir_neurons=5,
        connectivity=0.7,
        seed=42
    )

    # Generate reservoir states
    X_train_qrc = qrc.generate_reservoir_states(X_train_scaled)
    X_test_qrc  = qrc.generate_reservoir_states(X_test_scaled)

    # Use Logistic Regression as the "readout"
    qrc_readout = LogisticRegression(max_iter=1000)
    qrc_readout.fit(X_train_qrc, y_train)
    y_pred_qrc = qrc_readout.predict(X_test_qrc)

    acc_qrc = accuracy_score(y_test, y_pred_qrc)
    auc_qrc = roc_auc_score(y_test, y_pred_qrc)

    print(f"QRC Accuracy: {acc_qrc:.4f}")
    print(f"QRC ROC-AUC:  {auc_qrc:.4f}")

    # ----------------------------------------------------------------
    # 3.2 Logistic Regression Baseline
    # ----------------------------------------------------------------
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    print(f"Logistic Regression ROC-AUC:  {auc_lr:.4f}")

    # ----------------------------------------------------------------
    # 3.3 Random Forest Baseline
    # ----------------------------------------------------------------
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print(f"Random Forest ROC-AUC:  {auc_rf:.4f}")

    # ----------------------------------------------------------------
    # 3.4 SVM Baseline
    # ----------------------------------------------------------------
    print("\n=== SVM ===")
    svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    auc_svm = roc_auc_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {acc_svm:.4f}")
    print(f"SVM ROC-AUC:  {auc_svm:.4f}")

    # ----------------------------------------------------------------
    # 3.5 LSTM Baseline (single-timestep)
    # ----------------------------------------------------------------
    print("\n=== LSTM ===")
    # Reshape for LSTM: (samples, timesteps=1, features=5)
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm  = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    lstm_model = Sequential([
        LSTM(32, input_shape=(1, X_train_scaled.shape[1])),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    lstm_model.fit(X_train_lstm, y_train, 
                   epochs=10, batch_size=32, 
                   validation_split=0.1, verbose=0)

    y_pred_lstm_prob = lstm_model.predict(X_test_lstm)
    y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int)
    acc_lstm = accuracy_score(y_test, y_pred_lstm)
    auc_lstm = roc_auc_score(y_test, y_pred_lstm)

    print(f"LSTM Accuracy: {acc_lstm:.4f}")
    print(f"LSTM ROC-AUC:  {auc_lstm:.4f}")

    # ----------------------------------------------------------------
    # 3.6 Simple RNN (single-timestep)
    # ----------------------------------------------------------------
    print("\n=== Simple RNN ===")
    # Same reshape as LSTM
    X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_rnn  = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    rnn_model = Sequential([
        SimpleRNN(32, input_shape=(1, X_train_scaled.shape[1])),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    rnn_model.fit(X_train_rnn, y_train,
                  epochs=10, batch_size=32,
                  validation_split=0.1, verbose=0)

    y_pred_rnn_prob = rnn_model.predict(X_test_rnn)
    y_pred_rnn = (y_pred_rnn_prob > 0.5).astype(int)
    acc_rnn = accuracy_score(y_test, y_pred_rnn)
    auc_rnn = roc_auc_score(y_test, y_pred_rnn)

    print(f"Simple RNN Accuracy: {acc_rnn:.4f}")
    print(f"Simple RNN ROC-AUC:  {auc_rnn:.4f}")

    # ----------------------------------------------------------------
    # 4) Summary
    # ----------------------------------------------------------------
    print("\n=== Summary of All Models ===")
    print(f"QRC          -> Acc={acc_qrc:.4f}, AUC={auc_qrc:.4f}")
    print(f"LogReg       -> Acc={acc_lr:.4f},  AUC={auc_lr:.4f}")
    print(f"RandomForest -> Acc={acc_rf:.4f}, AUC={auc_rf:.4f}")
    print(f"SVM          -> Acc={acc_svm:.4f}, AUC={auc_svm:.4f}")
    print(f"LSTM         -> Acc={acc_lstm:.4f}, AUC={auc_lstm:.4f}")
    print(f"Simple RNN   -> Acc={acc_rnn:.4f}, AUC={auc_rnn:.4f}")


if __name__ == "__main__":
    main()