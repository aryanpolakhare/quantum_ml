import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

###############################################################################
# 1) Data Fetch & Preprocessing
###############################################################################
def fetch_stock_data(ticker="AAPL", start="2020-01-01", end="2024-01-01"):
    """
    Download stock data from Yahoo Finance and create a minimal target:
      Target = 1 if next day's close is higher than today's close, else 0.
    """
    print(f"\nFetching stock data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.dropna(inplace=True)
    
    # Minimal label: next day's close is higher or not
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    print(f"{ticker}: total records after preprocessing: {len(df)}")
    return df

def create_rolling_windows(df, feature_cols, window_size=5):
    """
    Returns X, y where:
      X.shape = (N, window_size, num_features),
      y.shape = (N,).
    Each sample is a rolling window of length 'window_size'. 
    The label is the last day in that window: df["Target"][i+window_size-1].
    """
    data = df[feature_cols].values
    targets = df["Target"].values
    N = len(df)

    X_list, y_list = [], []
    for i in range(N - window_size):
        X_window = data[i : i + window_size]
        y_val = targets[i + window_size - 1]  # label at the last day
        X_list.append(X_window)
        y_list.append(y_val)
    X_array = np.array(X_list)  # shape: (N-window_size, window_size, num_features)
    y_array = np.array(y_list)
    return X_array, y_array

###############################################################################
# 2) Quantum Reservoir Computing (QRC)
###############################################################################
class QuantumReservoir:
    """
    A toy quantum reservoir with random internal parameters:
      - n_qubits internal dimension (just a vector).
      - random "unitary" matrix + random input mapping.
      - no training of the internal parameters.
    """
    def __init__(self, n_qubits=4, input_dim=5, seed=42):
        np.random.seed(seed)
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        # Random "unitary"
        self.random_unitary = np.random.randn(n_qubits, n_qubits)
        # Random input map
        self.input_map = np.random.randn(input_dim, n_qubits)

    def reservoir_evolution(self, x_t, h_t):
        """
        x_t: shape (input_dim,)
        h_t: shape (n_qubits,)
        Return updated reservoir state after one time step.
        """
        # Input injection
        input_inject = x_t @ self.input_map  # (n_qubits,)
        # Combine with old state
        new_state = h_t + input_inject
        # Random "unitary" transform
        new_state = new_state @ self.random_unitary
        # Nonlinear
        new_state = np.tanh(new_state)
        return new_state

    def collect_states(self, X):
        """
        X: shape (batch_size, window_size, input_dim).
        Returns final states of shape (batch_size, n_qubits).
        """
        batch_size, window_size, _ = X.shape
        states = np.zeros((batch_size, self.n_qubits), dtype=np.float32)

        for i in range(batch_size):
            h_t = np.zeros((self.n_qubits,))
            for t in range(window_size):
                x_t = X[i, t, :]
                h_t = self.reservoir_evolution(x_t, h_t)
            states[i] = h_t
        return states

def qrc_pipeline(X_train, y_train, X_test, y_test, readout_type="LR"):
    """
    1) Build a QuantumReservoir
    2) Collect states
    3) Train readout (LR or MLP)
    4) Evaluate on test
    """
    n_qubits = 4
    input_dim = X_train.shape[-1]
    reservoir = QuantumReservoir(n_qubits=n_qubits, input_dim=input_dim, seed=42)

    X_train_res = reservoir.collect_states(X_train)  # (N_train, n_qubits)
    X_test_res  = reservoir.collect_states(X_test)   # (N_test, n_qubits)

    if readout_type == "LR":
        clf = LogisticRegression()
        clf.fit(X_train_res, y_train)
        prob_test = clf.predict_proba(X_test_res)[:, 1]
    else:
        # MLP
        model = models.Sequential([
            layers.Input(shape=(n_qubits,)),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train_res, y_train, epochs=10, batch_size=32, verbose=0)
        prob_test = model.predict(X_test_res).ravel()

    y_pred = (prob_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, prob_test)
    return acc, auc

###############################################################################
# 3) Quantum Liquid Neural Network (Q-LNN)
###############################################################################
class QuantumLNNCell(layers.Layer):
    """
    A toy custom RNN cell with "quantum" parameters. 
    Everything is trainable: input->angles, recurrent->update, etc.
    """
    def __init__(self, n_qubits=4, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.state_size = n_qubits

    def build(self, input_shape):
        # input_shape: (batch_size, input_dim)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.n_qubits),
            initializer="random_normal",
            trainable=True,
            name="kernel"
        )
        self.bias = self.add_weight(
            shape=(self.n_qubits,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.n_qubits, self.n_qubits),
            initializer="orthogonal",
            trainable=True,
            name="recurrent_kernel"
        )
        super().build(input_shape)

    def call(self, inputs, states):
        # inputs: (batch_size, input_dim)
        prev_state = states[0]  # (batch_size, n_qubits)
        angles = tf.matmul(inputs, self.kernel) + self.bias
        # Toy "quantum" transform
        quantum_output = tf.math.sin(angles)
        # Recurrent update
        new_state = tf.matmul(prev_state, self.recurrent_kernel) + quantum_output
        new_state = tf.nn.tanh(new_state)
        return new_state, [new_state]

def build_q_lnn_model(input_dim, n_qubits=4):
    """
    RNN(QuantumLNNCell) -> Dense(1)
    """
    inp = layers.Input(shape=(None, input_dim))
    x = layers.RNN(QuantumLNNCell(n_qubits=n_qubits), return_sequences=False)(inp)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="Q_LNN")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def q_lnn_pipeline(X_train, y_train, X_test, y_test, epochs=10):
    input_dim = X_train.shape[-1]
    model = build_q_lnn_model(input_dim=input_dim, n_qubits=4)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    prob_test = model.predict(X_test).ravel()
    y_pred = (prob_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, prob_test)
    return acc, auc

###############################################################################
# 4) Main: Run experiment on multiple tickers
###############################################################################
def main():
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]  # you can add more
    start_date = "2020-01-01"
    end_date   = "2024-01-01"
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    window_size = 5

    # We'll store results in a dictionary to print later
    results = []

    for ticker in tickers:
        df = fetch_stock_data(ticker, start=start_date, end=end_date)
        if len(df) < window_size:
            print(f"{ticker} does not have enough data; skipping.")
            continue

        # Create rolling windows
        X, y = create_rolling_windows(df, feature_cols, window_size=window_size)
        
        # Minimal standard scaling of raw columns
        orig_shape = X.shape
        X_2d = X.reshape(-1, len(feature_cols))  # flatten to 2D
        scaler = StandardScaler()
        X_2d_scaled = scaler.fit_transform(X_2d)
        X_scaled = X_2d_scaled.reshape(orig_shape)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False
        )

        #######################################################################
        # QRC + LR
        #######################################################################
        acc_qrc_lr, auc_qrc_lr = qrc_pipeline(
            X_train, y_train, X_test, y_test, readout_type="LR"
        )

        #######################################################################
        # QRC + MLP
        #######################################################################
        acc_qrc_mlp, auc_qrc_mlp = qrc_pipeline(
            X_train, y_train, X_test, y_test, readout_type="MLP"
        )

        #######################################################################
        # Q-LNN
        #######################################################################
        acc_qlnn, auc_qlnn = q_lnn_pipeline(X_train, y_train, X_test, y_test, epochs=10)

        #######################################################################
        # Store results
        #######################################################################
        results.append({
            "Ticker": ticker,
            "QRC+LR_Acc": acc_qrc_lr,   "QRC+LR_AUC": auc_qrc_lr,
            "QRC+MLP_Acc": acc_qrc_mlp,"QRC+MLP_AUC": auc_qrc_mlp,
            "Q-LNN_Acc": acc_qlnn,     "Q-LNN_AUC": auc_qlnn,
        })

    # Print summary table
    print("\n==== Final Results (No Extra Feature Engineering) ====\n")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
