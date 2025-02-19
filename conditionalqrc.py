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
        raise ValueError("No data returned. Check ticker/date range or spelling of the ticker.")

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
    X = df[features].values  # shape: (num_samples, 5)
    y = df["Trend"].values   # shape: (num_samples,)

    print(f"Total records after preprocessing: {len(df)}")
    return X, y


# ---------------------------------------------------------------------
# 2) CONDITIONAL QUANTUM RESERVOIR COMPUTING CLASS
# ---------------------------------------------------------------------
class ConditionalQRC:
    """
    A simple example of a "Conditional" Quantum Reservoir Computer (CQRC).
    - We have n_qubits = 6
    - We have an internal reservoir of dimension = 8
    - The input dimension is 5 features (X has shape (N,5))
    
    The "condition" we use is whether the average reservoir state is >= 0 or < 0.
    If the average reservoir state >= 0, we apply a certain set of entangling gates.
    Otherwise, we apply a different set (or skip them entirely).
    """

    def __init__(self, n_qubits=6, n_reservoir_neurons=8, connectivity=0.7, seed=42):
        self.n_qubits = n_qubits
        self.n_reservoir_neurons = n_reservoir_neurons
        self.connectivity = connectivity

        self.n_features = 5  # We have 5 features

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        np.random.seed(seed)

        # Reservoir weight matrix: shape (8,5) to match 5 input features
        base_weights = np.random.normal(
            0, 1 / np.sqrt(self.n_features),
            size=(self.n_reservoir_neurons, self.n_features)
        )
        mask = (np.random.rand(self.n_reservoir_neurons, self.n_features) < connectivity)
        self.reservoir_weights = base_weights * mask

    def quantum_layer_conditional(self, inputs, condition):
        """
        A quantum circuit that:
          - Encodes 'inputs' (size = n_qubits or smaller) as RY rotations
          - Then applies EITHER:
               Entangling pattern A, if condition = True
               Entangling pattern B (or no entangling), if condition = False
          - Returns the expectation values of PauliZ on each qubit
        """
        @qml.qnode(self.dev)
        def circuit(x):
            # x is length = n_qubits
            for i in range(self.n_qubits):
                qml.RY(x[i] * np.pi, wires=i)

            # Condition-based entangling
            if condition:
                # e.g. entangle every pair
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CNOT(wires=[i, j])
            else:
                # e.g. entangle in a linear chain only
                # (or skip entirely, or do some other pattern)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit(inputs)

    def generate_reservoir_states(self, X):
        """
        For each sample (5D):
          1) project to 8D via reservoir_weights -> tanh
          2) condition = (mean of reservoir state >= 0)?
          3) feed first 6 dims into quantum_layer_conditional w/ that condition
          4) output shape = (6,)
        Returns array (N, 6).
        """
        reservoir_outputs = []
        for sample in X:
            # 5D -> 8D reservoir
            proj = np.dot(self.reservoir_weights, sample)  # (8,)
            proj = np.tanh(proj)  # still (8,)

            # condition: mean >= 0
            condition = (proj.mean() >= 0)

            # slice to 6 for the quantum circuit
            circuit_input = proj[: self.n_qubits]

            # run quantum circuit with conditional entangling pattern
            q_out = self.quantum_layer_conditional(circuit_input, condition)
            reservoir_outputs.append(q_out)

        return np.array(reservoir_outputs)  # shape = (N,6)


# ---------------------------------------------------------------------
# 3) MAIN: TRAIN/COMPARE CQRC & CLASSICAL MODELS
# ---------------------------------------------------------------------
def main():
    # Ask user which ticker to use
    ticker = input("Enter the stock ticker for prediction (e.g., 'AAPL'): ")
    if not ticker:
        ticker = "AAPL"

    start_date = "2020-01-01"
    end_date   = "2024-01-01"

    # 1. Fetch data
    X, y = fetch_and_preprocess_data(ticker, start_date, end_date)

    # 2. Train/Test Split (chronologically)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ----------------------------------------------------------------
    # 3.1 Conditional Quantum Reservoir Computing
    # ----------------------------------------------------------------
    print("\n=== Conditional Quantum Reservoir Computing (CQRC) ===")
    cqrc = ConditionalQRC(
        n_qubits=6,
        n_reservoir_neurons=8,
        connectivity=0.7,
        seed=42
    )

    # Generate reservoir states
    X_train_cqrc = cqrc.generate_reservoir_states(X_train_scaled)  # (train_samples, 6)
    X_test_cqrc  = cqrc.generate_reservoir_states(X_test_scaled)   # (test_samples, 6)

    # Use Logistic Regression as the readout
    cqrc_readout = LogisticRegression(max_iter=1000)
    cqrc_readout.fit(X_train_cqrc, y_train)
    y_pred_cqrc = cqrc_readout.predict(X_test_cqrc)

    acc_cqrc = accuracy_score(y_test, y_pred_cqrc)
    auc_cqrc = roc_auc_score(y_test, y_pred_cqrc)

    print(f"CQRC Accuracy: {acc_cqrc:.4f}")
    print(f"CQRC ROC-AUC:  {auc_cqrc:.4f}")

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

    lstm_model.fit(
        X_train_lstm, y_train,
        epochs=10, batch_size=32,
        validation_split=0.1, verbose=0
    )

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
    # Same reshape as for LSTM
    X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_rnn  = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    rnn_model = Sequential([
        SimpleRNN(32, input_shape=(1, X_train_scaled.shape[1])),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    rnn_model.fit(
        X_train_rnn, y_train,
        epochs=10, batch_size=32,
        validation_split=0.1, verbose=0
    )

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
    print(f"Conditional QRC -> Acc={acc_cqrc:.4f}, AUC={auc_cqrc:.4f}")
    print(f"Logistic Reg    -> Acc={acc_lr:.4f},   AUC={auc_lr:.4f}")
    print(f"RandomForest    -> Acc={acc_rf:.4f},  AUC={auc_rf:.4f}")
    print(f"SVM             -> Acc={acc_svm:.4f}, AUC={auc_svm:.4f}")
    print(f"LSTM            -> Acc={acc_lstm:.4f}, AUC={auc_lstm:.4f}")
    print(f"Simple RNN      -> Acc={acc_rnn:.4f}, AUC={auc_rnn:.4f}")


if __name__ == "__main__":
    main()
