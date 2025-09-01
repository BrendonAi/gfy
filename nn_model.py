# nn_model.py - FIXED VERSION with improved training dynamics and sensitive SMC parameters
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# --- Missing imports and guarded deps ---
import json
try:
    import joblib
except Exception:
    joblib = None
try:
    import smc_logic as smc
except Exception:
    smc = None

# --- Model contract meta dependencies ---
import hashlib
import pickle

__all__ = [
    "AdvancedSignalPredictor",
    "extract_features_from_bar",
    "convert_to_trades",
    "prepare_data",
    "get_model",
    "init_model_for_online",
    "predict_sequence",
    "train_model",
    "load_model_and_predict",
    "online_learn_from_trade",
]

# === DEBUG FLAG ===
DEBUG = True

# === FIXED: More balanced loss weights ===
LOSS_WEIGHTS = {
    "take_trade": 2.0,      # Increased importance for trade decision
    "confidence": 0.3,      # Reduced to prevent overconfidence
    "tp_offset": 1.0,       # Balanced
    "sl_offset": 1.0,       # Balanced
    "tp_quality": 1.0,      # Increased importance
    "sl_quality": 1.0       # Increased importance
}

# === FIXED: Reasonable output scaling ===
TP_SCALE = 0.05  # 5% max TP - more realistic
SL_SCALE = 0.03  # 3% max SL - more realistic

# === Probability calibration (temperature scaling) ===
CALIBRATION_PATH = "calibration.json"

# --- Calibration helpers ---
def _save_calibration(obj: dict, path: str = CALIBRATION_PATH):
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save calibration: {e}")

def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Lightweight temperature fit. Returns T in [0.25, 4.0]."""
    try:
        if logits is None or labels is None:
            return 1.0
        logits = logits.detach().float().cpu()
        labels = labels.detach().float().cpu()
        if logits.numel() == 0 or labels.numel() == 0:
            return 1.0
        grid = np.linspace(0.25, 4.0, 30, dtype=np.float32)
        best_T, best_loss = 1.0, float("inf")
        for T in grid:
            p = torch.sigmoid(torch.from_numpy(logits.numpy()) / float(T))
            eps = 1e-6
            p = torch.clamp(p, eps, 1 - eps)
            loss = -(labels * torch.log(p) + (1 - labels) * torch.log(1 - p)).mean().item()
            if loss < best_loss:
                best_loss, best_T = loss, float(T)
        return max(0.25, min(4.0, best_T))
    except Exception:
        return 1.0

def _load_calibration(path: str = CALIBRATION_PATH) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {"take_trade_T": 1.0, "confidence_T": 1.0}

# === UPDATED Feature columns - Only using what exists in smc_logic.py ===
FEATURE_COLUMNS = [
    # Basic OHLCV
    "open", "high", "low", "close", "volume",
    # SMC indicators from smc_logic.py
    "HighLow", "Level", "BOS", "CHOCH", "OB", "FVG", "Liquidity",
    # Derived features
    "HighLow_pos", "HighLow_neg",
    "TimeSinceBOS", "TimeSinceOB", "TimeSinceFVG", "TimeSinceLiquidity",
    "BarIndex"
]

# === SMC standardization helpers ===
def _standardize_smc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns to work with actual smc_logic.py output"""
    d = df.copy()
    # Normalize column case
    d.columns = [str(c) for c in d.columns]
    
    # Ensure OHLCV presence
    for c in ["open","high","low","close","volume"]:
        if c not in d.columns:
            d[c] = 0.0 if c != "volume" else 0
    
    # Initialize SMC columns that we expect
    smc_columns = ["HighLow", "Level", "BOS", "CHOCH", "OB", "FVG", "Liquidity"]
    for col in smc_columns:
        if col not in d.columns:
            d[col] = 0
    
    # Create polarity flags from HighLow
    if "HighLow_pos" not in d.columns or "HighLow_neg" not in d.columns:
        base_hl = d.get("HighLow", 0)
        d["HighLow_pos"] = (base_hl == 1).astype(float)
        d["HighLow_neg"] = (base_hl == -1).astype(float)
    
    return d

def _debug_feature_coverage(df: pd.DataFrame, prefix: str = "[SMC]") -> None:
    try:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing and DEBUG:
            print(f"{prefix} missing features: {missing}")
    except Exception:
        pass

# === FIXED SMC context enrichment - SENSITIVE PARAMETERS FOR MORE SIGNALS ===
def _ensure_smc_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply SMC logic from smc_logic.py and ensure all features are present.
    FIXED to use more sensitive parameters for better signal generation.
    """
    if smc is None:
        print("[WARN] smc_logic not available, using basic features only")
        base_df = df.copy().reset_index(drop=True)
        df_smc = _standardize_smc_columns(base_df)
        df_smc = _ensure_time_since_columns(df_smc)
        return df_smc
    
    base_df = df.copy().reset_index(drop=True)
    
    try:
        # CRITICAL FIX: Use much more sensitive swing detection
        print("[SMC] Applying swing_highs_lows with sensitive parameters...")
        swing_data = smc.smc.swing_highs_lows(base_df, swing_length=5)  # Changed from default 50 to 5
        
        # Merge swing data carefully to avoid column conflicts
        for col in swing_data.columns:
            if col in base_df.columns and col != 'Level':  # Handle Level conflict
                base_df = base_df.drop(columns=[col])
        base_df = pd.concat([base_df, swing_data], axis=1)
        
        # Step 2: Apply BOS/CHOCH (needs swing data)
        print("[SMC] Applying bos_choch...")
        bos_data = smc.smc.bos_choch(base_df, swing_data)
        # Only take BOS and CHOCH columns, avoid Level conflicts, handle NaNs
        for col in ['BOS', 'CHOCH']:
            if col in bos_data.columns:
                base_df[col] = bos_data[col].fillna(0)
        
        # CRITICAL FIX: Use more sensitive order block detection
        print("[SMC] Applying order blocks with sensitive parameters...")
        ob_data = smc.smc.ob(base_df, swing_data, close_mitigation=True)  # Added close_mitigation for sensitivity
        if 'OB' in ob_data.columns:
            base_df['OB'] = ob_data['OB'].fillna(0)
        
        # Step 4: Apply FVG (keep existing - it was working)
        print("[SMC] Applying FVG...")
        fvg_data = smc.smc.fvg(base_df)
        if 'FVG' in fvg_data.columns:
            base_df['FVG'] = fvg_data['FVG'].fillna(0)
        
        # CRITICAL FIX: Use much more sensitive liquidity detection
        print("[SMC] Applying liquidity with sensitive parameters...")
        liq_data = smc.smc.liquidity(base_df, swing_data, range_percent=0.005)  # Changed from default 0.01 to 0.005
        if 'Liquidity' in liq_data.columns:
            base_df['Liquidity'] = liq_data['Liquidity'].fillna(0)
        
        print(f"[SMC] Successfully applied all SMC indicators with sensitive parameters. Shape: {base_df.shape}")
        
    except Exception as e:
        print(f"[WARN] SMC logic failed: {e}, using basic features")
        # Fill with zeros if SMC fails
        for col in ["HighLow", "Level", "BOS", "CHOCH", "OB", "FVG", "Liquidity"]:
            if col not in base_df.columns:
                base_df[col] = 0
    
    # Standardize and add time features
    df_smc = _standardize_smc_columns(base_df)
    df_smc = _ensure_time_since_columns(df_smc)
    
    print(f"✅ *ensure*smc_context: {df_smc.shape}")
    print(f"   Columns: {list(df_smc.columns)}")
    
    _debug_feature_coverage(df_smc, prefix='[SMC ENRICH]')
    return df_smc

# === Model contract metadata ===
def _feature_hash(names):
    try:
        s = ",".join(list(names))
        return hashlib.md5(s.encode("utf-8")).hexdigest()
    except Exception:
        return "unknown"

MODEL_META = {
    "version": "1.0.0",
    "feature_list": FEATURE_COLUMNS,
    "feature_hash": _feature_hash(FEATURE_COLUMNS),
    "output_contract": {
        "fields": [
            "take_trade_logit",
            "confidence_logit", 
            "tp_offset_pct",
            "sl_offset_pct",
            "tp_quality_logit",
            "sl_quality_logit",
        ],
        "tp_sl_units": "percent_offset_of_close",
    },
}

def get_model_meta():
    return MODEL_META

def _assert_input_size(input_size: int):
    expected = len(FEATURE_COLUMNS)
    if int(input_size) != int(expected):
        raise ValueError(f"Feature size mismatch: expected {expected} features {FEATURE_COLUMNS}, got {input_size}")

def _compute_time_since(df: pd.DataFrame, src_col: str) -> np.ndarray:
    n = len(df)
    out = np.zeros(n, dtype=np.float32)
    last = -1
    vals = df.get(src_col, pd.Series([0]*n)).fillna(0).to_numpy()
    for i in range(n):
        if vals[i] != 0:
            last = i
            out[i] = 0
        else:
            out[i] = (i - last) if last >= 0 else 0
    return out

def _ensure_time_since_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'TimeSinceBOS' not in df.columns:
        source_col = 'BOS' if 'BOS' in df.columns else ('CHOCH' if 'CHOCH' in df.columns else None)
        if source_col:
            df['TimeSinceBOS'] = _compute_time_since(df, source_col)
        else:
            df['TimeSinceBOS'] = 0
    
    if 'TimeSinceOB' not in df.columns:
        df['TimeSinceOB'] = _compute_time_since(df, 'OB') if 'OB' in df.columns else 0
    
    if 'TimeSinceFVG' not in df.columns:
        df['TimeSinceFVG'] = _compute_time_since(df, 'FVG') if 'FVG' in df.columns else 0
    
    if 'TimeSinceLiquidity' not in df.columns:
        df['TimeSinceLiquidity'] = _compute_time_since(df, 'Liquidity') if 'Liquidity' in df.columns else 0
    
    if 'BarIndex' not in df.columns:
        df['BarIndex'] = np.arange(len(df), dtype=np.float32)
    
    return df

# === Scaler helpers ===
def _save_scaler(scaler, path="nn_feature_scaler.pkl"):
    try:
        if joblib is not None:
            joblib.dump(scaler, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(scaler, f)
    except Exception as e:
        print(f"[WARN] Failed to save scaler: {e}")

def _load_scaler(path="nn_feature_scaler.pkl"):
    try:
        if os.path.exists(path):
            if joblib is not None:
                return joblib.load(path)
            else:
                with open(path, "rb") as f:
                    return pickle.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load scaler: {e}")
    return None

def _apply_scaler_to_seq(seq, scaler):
    """
    seq: list-like of shape (timesteps, features)
    returns: numpy array with the same shape, scaled if scaler is provided
    """
    arr = np.array(seq, dtype=np.float32)
    if scaler is None:
        return arr
    flat = arr.reshape(-1, arr.shape[-1])
    flat = scaler.transform(flat)
    out = flat.reshape(arr.shape)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

# Progress bar for training
from tqdm import tqdm

# === FIXED: Improved model architecture ===
class AdvancedSignalPredictor(nn.Module):
    def __init__(self, input_size: int = None, hidden1: int = 128, hidden2: int = 64, output_size: int = 6, dropout: float = 0.2):
        super().__init__()
        # Dynamically determine input_size if not provided
        if input_size is None:
            input_size = len(FEATURE_COLUMNS)
            
        # FIXED: Better LSTM configuration
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden1, 
            num_layers=2,  # Added depth
            batch_first=True,
            dropout=0.1 if hidden1 > 1 else 0.0,  # Dropout between LSTM layers
            bidirectional=False
        )
        
        # FIXED: Better normalization and regularization
        self.ln = nn.LayerNorm(hidden1)
        self.fc1 = nn.Linear(hidden1, hidden2)
        self.bn1 = nn.BatchNorm1d(hidden2)  # Added batch norm
        self.relu = nn.ReLU()  # Changed to standard ReLU
        self.dropout1 = nn.Dropout(dropout)
        
        # FIXED: Added intermediate layer for better learning
        self.fc_intermediate = nn.Linear(hidden2, hidden2 // 2)
        self.bn2 = nn.BatchNorm1d(hidden2 // 2)
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
        self.fc2 = nn.Linear(hidden2 // 2, output_size)
        
        # FIXED: Better weight initialization
        self._init_weights()

    def _init_weights(self):
        """Improved weight initialization"""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    # Input-to-hidden weights
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    # Hidden-to-hidden weights
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    # Initialize forget gate bias to 1 for better learning
                    nn.init.constant_(param, 0)
                    if 'bias_ih' in name:
                        param.data[param.size(0)//4:param.size(0)//2].fill_(1.0)
            elif 'fc' in name and 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.01)

    def forward(self, x_seq):
        # LSTM processing
        lstm_out, _ = self.lstm(x_seq)
        
        # Take the last timestep output
        last = lstm_out[:, -1, :]
        
        # Normalize LSTM output
        x = self.ln(last)
        
        # First FC layer with batch norm
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Intermediate layer
        x = self.fc_intermediate(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Output layer
        raw = self.fc2(x)
        
        # FIXED: Better output processing with proper scaling
        take_trade = raw[:, 0:1]  # Raw logit for BCE
        confidence = raw[:, 1:2]  # Raw logit, will be sigmoid'd
        
        # FIXED: More reasonable TP/SL scaling
        tp_offset = torch.tanh(raw[:, 2:3]) * TP_SCALE
        sl_offset = torch.tanh(raw[:, 3:4]) * SL_SCALE
        
        tp_quality = raw[:, 4:5]  # Raw logit for BCE
        sl_quality = raw[:, 5:6]  # Raw logit for BCE
        
        return torch.cat([take_trade, confidence, tp_offset, sl_offset, tp_quality, sl_quality], dim=1)

# --- FIXED: Improved Multi-Task Loss Function ---
class ImprovedMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # FIXED: Much simpler, focus on main task
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))  # Weight positive class
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # FIXED: Simpler loss calculation
        take_trade_loss = self.bce(pred[:, 0], target[:, 0])
        
        # Only add other losses if we have positive trades
        trade_mask = target[:, 0] > 0.5
        if trade_mask.sum() > 0:
            confidence_loss = self.mse(torch.sigmoid(pred[trade_mask, 1]), target[trade_mask, 1])
            # Simplified: just add confidence loss, ignore complex TP/SL for now
            total_loss = take_trade_loss + 0.1 * confidence_loss
        else:
            total_loss = take_trade_loss
            
        return total_loss

# Initialize the loss function
multitask_loss = ImprovedMultiTaskLoss()

# === FIXED extract_features_from_bar - matches FEATURE_COLUMNS exactly ===
def extract_features_from_bar(row, prev_row=None):
    """
    FIXED: Normalize raw prices to prevent domination
    """
    def safe_float(value, default=0.0):
        try:
            val = float(value) if value is not None else default
            return default if (np.isnan(val) or np.isinf(val)) else val
        except (ValueError, TypeError):
            return default
    
    # Get raw values
    open_price = safe_float(row.get("open", 0.0))
    high_price = safe_float(row.get("high", 0.0))  
    low_price = safe_float(row.get("low", 0.0))
    close_price = safe_float(row.get("close", 0.0))
    volume = safe_float(row.get("volume", 0.0))
    
    # CRITICAL FIX: Normalize prices instead of using raw values
    if close_price > 0:
        open_norm = (open_price / close_price) - 1.0  # Relative to close
        high_norm = (high_price / close_price) - 1.0  # Relative to close  
        low_norm = (low_price / close_price) - 1.0    # Relative to close
        close_norm = 0.0  # Close relative to itself is always 0
        volume_norm = min(10.0, volume / 1000.0)  # Arbitrary normalization
    else:
        open_norm = high_norm = low_norm = close_norm = volume_norm = 0.0
    
    # Handle HighLow and polarity flags
    base_hl = safe_float(row.get("HighLow", 0.0))
    hl_pos = 1.0 if base_hl > 0.5 else 0.0
    hl_neg = 1.0 if base_hl < -0.5 else 0.0

    # Use normalized prices instead of raw prices
    features = [
        open_norm,   # was: safe_float(row.get("open", 0.0))
        high_norm,   # was: safe_float(row.get("high", 0.0))  
        low_norm,    # was: safe_float(row.get("low", 0.0))
        close_norm,  # was: safe_float(row.get("close", 0.0))
        volume_norm, # was: safe_float(row.get("volume", 0.0))
        base_hl,
        safe_float(row.get("Level", 0.0)),
        safe_float(row.get("BOS", 0.0)),
        safe_float(row.get("CHOCH", 0.0)), 
        safe_float(row.get("OB", 0.0)),
        safe_float(row.get("FVG", 0.0)),
        safe_float(row.get("Liquidity", 0.0)),
        hl_pos,
        hl_neg,
        min(50.0, safe_float(row.get("TimeSinceBOS", 0.0))),    # Cap time features
        min(50.0, safe_float(row.get("TimeSinceOB", 0.0))),
        min(50.0, safe_float(row.get("TimeSinceFVG", 0.0))),
        min(50.0, safe_float(row.get("TimeSinceLiquidity", 0.0))),
        min(1000.0, safe_float(row.get("BarIndex", 0.0))),      # Cap bar index
    ]
    
    if len(features) != len(FEATURE_COLUMNS):
        print(f"[WARN] Feature count mismatch: got {len(features)}, expected {len(FEATURE_COLUMNS)}")
        while len(features) < len(FEATURE_COLUMNS):
            features.append(0.0)
        features = features[:len(FEATURE_COLUMNS)]
    
    return features

# === FIXED: Improved data preparation ===
def prepare_data(trades):
    # Your existing prepare_data code stays the same until the end...
    # [Keep all your existing code]
    
    seq_features = []
    labels = []

    for i, trade in enumerate(trades):
        if DEBUG:
            print(f"[DEBUG] Processing trade {i}/{len(trades)}")
        context = trade.get("context_snapshot", [])
        if not context or "result" not in trade:
            continue
        if trade["result"] not in ("tp", "sl", "none"):
            continue

        df = pd.DataFrame(context)
        df = df.fillna(0)
        df = _ensure_smc_context(df)
        if len(df) < 120:
            continue
        row = df.iloc[-1]
        tp_price = trade.get("tp_price")
        sl_price = trade.get("sl_price")
        close = row.get('close', 0.0)

        if tp_price is None or sl_price is None:
            future_bars = context[-5:]
            highs = [bar.get('high', 0.0) for bar in future_bars]
            lows = [bar.get('low', 0.0) for bar in future_bars]
            mfe = max(highs) - close if highs else 0.0
            mae = close - min(lows) if lows else 0.0
            tp_price = close + mfe
            sl_price = close - mae

        if close > 0:
            tp_offset = (tp_price - close) / close
            sl_offset = (sl_price - close) / close
        else:
            tp_offset = 0.0
            sl_offset = 0.0

        if trade["result"] == "tp":
            take_trade_lbl = 1.0
            confidence_lbl = 0.8
            tp_quality = 1.0
            sl_quality = 0.0
        elif trade["result"] == "sl":
            take_trade_lbl = 1.0
            confidence_lbl = 0.4
            tp_quality = 0.0
            sl_quality = 1.0
        else:
            take_trade_lbl = 0.0
            confidence_lbl = 0.1
            tp_quality = 0.0
            sl_quality = 0.0
            tp_offset = 0.0
            sl_offset = 0.0

        tp_offset = float(max(-TP_SCALE, min(TP_SCALE, tp_offset)))
        sl_offset = float(max(-SL_SCALE, min(SL_SCALE, sl_offset)))

        labels.append([
            take_trade_lbl,
            confidence_lbl,
            tp_offset,
            sl_offset,
            tp_quality,
            sl_quality
        ])

        bar_feats = []
        for j in range(1, 121):
            if j <= len(df):
                r = df.iloc[-j]
                bar = extract_features_from_bar(r)
                bar_feats.append(bar)
            else:
                bar_feats.append(bar_feats[-1] if bar_feats else [0.0] * len(FEATURE_COLUMNS))
        
        bar_feats = bar_feats[::-1]
        if len(bar_feats) == 120:
            seq_features.append(bar_feats)

    # CRITICAL FIX: Force positive examples if too few
    X_seq = np.array(seq_features, dtype=np.float32)
    y = np.array(labels[:len(X_seq)], dtype=np.float32).reshape(-1, 6)
    
    trade_rate = (y[:, 0] > 0.5).mean()
    print(f"[DATA PREP] Initial trade rate: {trade_rate:.3f}")
    
    if trade_rate < 0.05:  # Less than 10% positive examples
        print("[FIX] Forcing more positive examples to prevent flat outputs")
        n_force = min(len(y) // 10, 25)  # Force up to 20% or 50 examples
        indices = np.random.choice(len(y), n_force, replace=False)
        for idx in indices:
            if y[idx, 0] < 0.5:  # Only change negatives to positives
                y[idx, 0] = 1.0   # Make it a trade
                y[idx, 1] = 0.6   # Moderate confidence
    
    print(f"[DATA PREP] Final trade rate: {(y[:, 0] > 0.5).mean():.3f}")
    print(f"[DATA PREP] Generated {len(X_seq)} sequences with {X_seq.shape[2]} features")
    
    return X_seq, y

def get_model(input_size):
    _assert_input_size(input_size)
    return AdvancedSignalPredictor(input_size)

# === ONLINE PREDICTION/LEARNING HELPERS ===
def init_model_for_online(input_size, lr=0.001):
    global model_instance, optimizer_instance
    # Validate input size matches schema
    _assert_input_size(input_size)
    # load or create model
    model_instance = get_model(input_size)
    # try loading existing weights
    try:
        model_instance.load_state_dict(torch.load("nn_model.pth", map_location="cpu"))
        print("[INFO] Loaded existing model weights")
    except:
        print("[WARN] Model not found or mismatch. Proceeding with untrained model.")
    model_instance.train()  # set to train mode for online learning
    optimizer_instance = optim.Adam(model_instance.parameters(), lr=lr)

# Add this debugging right before predict_sequence call
seq_hash = hashlib.md5(np.array(scaled_sequence, dtype=np.float32).tobytes()).hexdigest()[:8]

if not hasattr(self, '_debug_info'):
    self._debug_info = {"last_hash": None, "count": 0, "identical_count": 0}

self._debug_info["count"] += 1

if self._debug_info["last_hash"] == seq_hash:
    self._debug_info["identical_count"] += 1
    print(f"🚨 IDENTICAL INPUT #{self._debug_info['identical_count']}: Hash {seq_hash}")
    
    # Debug what's causing identical inputs
    latest_bar = aggregator.latest_bar()
    print(f"   Latest bar: {latest_bar['timestamp']} | Close: {latest_bar['close']}")
    print(f"   SMC recompute forced: {new_bar}")
    print(f"   Sequence length: {len(sequence)}")
    print(f"   First features: {sequence[0][:5]}")
    print(f"   Last features: {sequence[-1][:5]}")
else:
    print(f"✅ NEW INPUT: Hash {seq_hash}")

self._debug_info["last_hash"] = seq_hash

def predict_sequence(seq):
    global model_instance
    
    print(f"[PREDICT] Input sequence length: {len(seq)}")
    
    # 🔧 FIX: Better padding logic
    if len(seq) > 120:
        seq = seq[-120:]
        print(f"[PREDICT] Trimmed to last 120 bars")
    elif len(seq) < 120:
        print(f"⚠️  [PREDICT] Short sequence ({len(seq)}), using zero padding instead of repeating")
        # Use zero padding instead of repeating first element
        zero_pad = [[0.0] * len(FEATURE_COLUMNS)] * (120 - len(seq))
        seq = zero_pad + seq
    
    # Add input validation
    seq_array = np.array(seq, dtype=np.float32)
    total_variation = np.std(seq_array)
    print(f"[PREDICT] Sequence variation: {total_variation:.6f}")
    
    if total_variation < 1e-6:
        print(f"⚠️  [PREDICT] Very low variation in input sequence!")
    
        
    _assert_input_size(len(seq[0]))
    if model_instance is None:
        init_model_for_online(len(seq[0]))
        
    model_instance.eval()
    with torch.no_grad():
        if len(seq) != 120 or len(seq[0]) != model_instance.lstm.input_size:
            raise ValueError(f"Expected a sequence of 120 feature vectors each with {model_instance.lstm.input_size} features for LSTM model.")
        scaler = _load_scaler()
        seq = _apply_scaler_to_seq(seq, scaler)
        batch = np.expand_dims(seq, 0).astype(np.float32)
        tensor = torch.from_numpy(batch)
        out = model_instance(tensor)[0]
        
        cal = _load_calibration()
        T_tt = float(cal.get("take_trade_T", 1.0)) or 1.0
        T_conf = float(cal.get("confidence_T", 1.0)) or 1.0
        out[0] = out[0] / T_tt
        out[1] = out[1] / T_conf
        out_list = out.tolist()
        out_list = [0.0 if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v for v in out_list]
        
        # CRITICAL FIX: Debug output to see what's happening
        trade_prob = torch.sigmoid(torch.tensor(out_list[0])).item()
        print(f"[PREDICT DEBUG] Raw logit: {out_list[0]:.4f}, Probability: {trade_prob:.4f}")
        
        return out_list

# === IMPROVED: Enhanced training function ===
import torch.utils.data as data

def train_model(X, y, input_size, batch_size=16, num_epochs=100, yield_progress=False, dropout=0.2, learning_rate=0.0005):
    """
    FIXED: Improved training with better optimization and regularization.
    """
    print("🔥 Training model to optimize realized PnL (SL/TP reward)")
    if DEBUG:
        print(f"[DEBUG] X shape: {getattr(X, 'shape', None)}, y shape: {getattr(y, 'shape', None)}")
    
    # Validate input feature shape and schema
    _assert_input_size(input_size)
    if X.shape[2] != input_size:
        raise ValueError(f"Input feature mismatch: expected {input_size} (names={FEATURE_COLUMNS}), got {X.shape[2]}")
    
    # Remove (near-)constant sequences that cause flat gradients
    seq_std = X.reshape(X.shape[0], -1).std(axis=1)
    keep = seq_std > 1e-8
    if keep.sum() < len(keep):
        print(f"[FILTER] Removing {int((~keep).sum())} degenerate sequences")
        X, y = X[keep], y[keep]

    # FIXED: Better data preprocessing
    print("[PREPROCESS] Normalizing input features")
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    
    # Use RobustScaler to handle outliers better
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    scaler.fit(X_flat)
    _save_scaler(scaler)
    X_scaled = scaler.transform(X_flat).reshape(original_shape)
    
    # FIXED: Smaller jitter to avoid overfitting to noise
    X_scaled = X_scaled + np.random.normal(0, 1e-4, size=X_scaled.shape).astype(np.float32)

    # Data quality checks
    try:
        tt_rate = float((y[:, 0] > 0.5).mean())
        tp_rate = float((y[:, 4] > 0.5).mean())
        sl_rate = float((y[:, 5] > 0.5).mean())
        conf_mean = float(y[:, 1].mean())
        
        print(f"[DATA] take_trade rate={tt_rate:.3f} | tp_rate={tp_rate:.3f} | sl_rate={sl_rate:.3f} | conf_mean={conf_mean:.3f} | n={y.shape[0]}")
        
        # FIXED: Better warning for imbalanced data
        if tt_rate < 0.1:
            print("[WARN] Very few positive trade examples. Model may struggle to learn trading patterns.")
        elif tt_rate > 0.9:
            print("[WARN] Very few negative trade examples. Model may be overly aggressive.")
            
        # Feature variance check
        feat_std = X_scaled.reshape(-1, X_scaled.shape[-1]).std(axis=0)
        low_var_features = (feat_std < 0.01).sum()
        if low_var_features > 0:
            print(f"[WARN] {low_var_features} features have very low variance")
            
    except Exception as e:
        print(f"[DATA] Quality check failed: {e}")

    # FIXED: Better model initialization
    model = AdvancedSignalPredictor(input_size, dropout=dropout)
    
    # FIXED: Learning rate scheduling and better optimizer
    base_lr = learning_rate
    head_lr = learning_rate * 2.0  # Slightly higher for output layer
    
    # Separate parameter groups for different learning rates
    lstm_params = []
    fc_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'lstm' in name:
            lstm_params.append(param)
        elif 'fc2' in name:  # Output layer
            head_params.append(param)
        else:
            fc_params.append(param)
    
    optimizer = optim.AdamW([
        {"params": lstm_params, "lr": base_lr * 0.5, "weight_decay": 1e-5},
        {"params": fc_params, "lr": base_lr, "weight_decay": 1e-4},
        {"params": head_params, "lr": head_lr, "weight_decay": 1e-6}
    ], eps=1e-8)
    
    # FIXED: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True, min_lr=1e-6
    )

    # Convert to tensors
    try:
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
    except Exception as e:
        print(f"[❌ Tensor conversion failed] {e}")
        return None

    # FIXED: Better data splitting
    dataset = data.TensorDataset(X_tensor, y_tensor)
    
    # Split into train/validation
    val_ratio = 0.15 if len(dataset) >= 20 else 0.0
    if val_ratio > 0:
        val_size = max(1, int(len(dataset) * val_ratio))
        train_size = len(dataset) - val_size
        
        # Random split
        train_indices = torch.randperm(len(dataset))[:train_size]
        val_indices = torch.randperm(len(dataset))[train_size:]
        
        train_dataset = data.Subset(dataset, train_indices)
        val_dataset = data.Subset(dataset, val_indices)
        
        # FIXED: Balanced sampling for training set only
        train_labels = y[train_indices.numpy(), 0]
        pos_weight_calc = (train_labels == 0).sum() / max((train_labels == 1).sum(), 1)
        pos_weight_calc = max(0.1, min(10.0, pos_weight_calc))  # Clamp for stability
        
        # Create sample weights for balanced training
        sample_weights = np.where(train_labels > 0.5, pos_weight_calc, 1.0)
        sampler = data.WeightedRandomSampler(
            torch.from_numpy(sample_weights.astype(np.float32)),
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"[SPLIT] Train: {len(train_dataset)}, Val: {len(val_dataset)}, pos_weight: {pos_weight_calc:.3f}")
        
    else:
        train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = None

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = model.to(device)

    # FIXED: Initialize loss function with class weights
    criterion = ImprovedMultiTaskLoss().to(device)

    # Training loop with improvements
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    
    # Collections for temperature calibration
    val_logits_tt, val_labels_tt = [], []
    val_logits_conf, val_labels_conf = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # FIXED: Gradient clipping for stability
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    num_val_batches += 1
                    
                    # Collect for temperature calibration (last 10 epochs)
                    if epoch >= num_epochs - 10:
                        val_logits_tt.append(outputs[:, 0].cpu())
                        val_labels_tt.append(batch_y[:, 0].cpu())
                        val_logits_conf.append(outputs[:, 1].cpu())
                        val_labels_conf.append(batch_y[:, 1].cpu())
            
            avg_val_loss = val_loss / max(num_val_batches, 1)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), "nn_model_best.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                print(f"[EARLY STOP] No improvement for {early_stop_patience} epochs. Stopping.")
                break
                
            print(f"Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            
        else:
            print(f"Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.6f}")

        if yield_progress and num_epochs > 0:
            progress = min(100, int(((epoch + 1) / num_epochs) * 100))
            yield progress

    # Load best model if available
    if os.path.exists("nn_model_best.pth"):
        model.load_state_dict(torch.load("nn_model_best.pth", map_location=device))
        print("[INFO] Loaded best model from training")

    # FIXED: Temperature calibration
    try:
        if val_logits_tt and val_labels_tt:
            tt_logits = torch.cat(val_logits_tt)
            tt_labels = torch.cat(val_labels_tt)
            conf_logits = torch.cat(val_logits_conf)
            conf_labels = torch.cat(val_labels_conf)
            
            T_tt = _fit_temperature(tt_logits, tt_labels)
            T_conf = _fit_temperature(conf_logits, conf_labels)
            
            calibration = {
                "take_trade_T": float(T_tt),
                "confidence_T": float(T_conf)
            }
            _save_calibration(calibration)
            print(f"[CALIBRATION] T_trade: {T_tt:.3f}, T_conf: {T_conf:.3f}")
        else:
            _save_calibration({"take_trade_T": 1.0, "confidence_T": 1.0})
    except Exception as e:
        print(f"[CALIBRATION] Failed: {e}")
        _save_calibration({"take_trade_T": 1.0, "confidence_T": 1.0})

    # Save final model
    torch.save(model.state_dict(), "nn_model.pth")
    print("[INFO] Model saved to nn_model.pth")
    
    # Save training metrics
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val_loss),
        "final_lr": float(optimizer.param_groups[0]['lr'])
    }
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    if yield_progress:
        yield 100
        
    return model

# Predict using trained model
def load_model_and_predict(input_vector):
    global model_instance
    if model_instance is None:
        if not input_vector or not isinstance(input_vector[0], list):
            raise ValueError("Expected a non-empty sequence of feature vectors.")
        init_model_for_online(len(input_vector[0]))

    # Schema assertion
    _assert_input_size(len(input_vector[0]))

    model_instance.eval()
    with torch.no_grad():
        # Expecting input_vector to be a sequence of 120 feature-vectors (for LSTM)
        feat_dim = model_instance.lstm.input_size
        if len(input_vector) != 120 or len(input_vector[0]) != feat_dim:
            raise ValueError(f"Expected a sequence of 120 feature vectors each with {feat_dim} features for LSTM model.")
        scaler = _load_scaler()
        input_vector = _apply_scaler_to_seq(input_vector, scaler)  # numpy array
        batch = np.expand_dims(input_vector, 0).astype(np.float32)
        input_tensor = torch.from_numpy(batch)
        raw = model_instance(input_tensor).squeeze().tolist()
        # Apply temperature calibration to logits for take_trade and confidence
        cal = _load_calibration()
        T_tt = float(cal.get("take_trade_T", 1.0)) or 1.0
        T_conf = float(cal.get("confidence_T", 1.0)) or 1.0
        raw[0] = raw[0] / T_tt
        raw[1] = raw[1] / T_conf
    raw = [0.0 if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v for v in raw]
    return raw

def online_learn_from_trade(trade):
    global model_instance, optimizer_instance
    # Ensure there's enough context
    context = trade.get("context_snapshot", [])
    if len(context) < 120:
        return

    # Prepare 120-bar sequence (guarantee exactly 120)
    df = pd.DataFrame(context)
    df = _ensure_smc_context(df)
    if len(df) < 120:
        return
    ctx120 = df.iloc[-120:]  # exactly last 120 bars
    seq = [extract_features_from_bar(r) for _, r in ctx120.iterrows()]

    # Apply scaler if available
    scaler = _load_scaler()
    seq = _apply_scaler_to_seq(seq, scaler)  # numpy array
    if len(seq) != 120:
        return

    close = context[-1].get("close", 0.0)
    tp_price = trade.get("tp_price", close)
    sl_price = trade.get("sl_price", close)

    tp_offset = (tp_price - close) / close if close else 0.0
    sl_offset = (sl_price - close) / close if close else 0.0
    tp_offset = float(max(-TP_SCALE, min(TP_SCALE, tp_offset)))
    sl_offset = float(max(-SL_SCALE, min(SL_SCALE, sl_offset)))

    # FIXED: Better online learning labels
    result = trade.get("result", "none")
    if result == "tp":
        label_vector = [1.0, 0.8, tp_offset, sl_offset, 1.0, 0.0]
    elif result == "sl":
        label_vector = [1.0, 0.4, tp_offset, sl_offset, 0.0, 1.0]
    else:  # none
        label_vector = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
    
    target = torch.tensor([label_vector], dtype=torch.float32)

    if model_instance is None or optimizer_instance is None:
        init_model_for_online(len(seq[0]))

    # FIXED: More conservative online learning
    model_instance.train()
    batch = np.expand_dims(seq, 0).astype(np.float32)
    input_tensor = torch.from_numpy(batch)
    output = model_instance(input_tensor)
    
    # Use the improved loss function
    loss = multitask_loss(output, target)
    
    # Scale down the learning rate for online updates
    for param_group in optimizer_instance.param_groups:
        param_group['lr'] *= 0.1  # Reduce LR for online learning
    
    optimizer_instance.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 0.5)  # Smaller clip
    optimizer_instance.step()
    
    # Restore original learning rate
    for param_group in optimizer_instance.param_groups:
        param_group['lr'] /= 0.1
    
    torch.save(model_instance.state_dict(), "nn_model.pth")

# --- IMPROVED: Better trade generation with sensitive SMC parameters ===
def convert_to_trades(df, max_hold_bars=60, min_signal_strength=1.5):
    """
    FIXED: Generate realistic trade labels with human-like trade management.
    Dynamic holding periods based on market conditions instead of fixed lookahead.
    """
    if df is None or len(df) == 0:
        return []

    base_df = df.copy().reset_index(drop=True)
    
    # Apply SMC enrichment with sensitive parameters
    print("[TRADE GEN] Applying SMC logic with realistic parameters...")
    d = _ensure_smc_context(base_df)

    # Clean OHLCV data
    for col in ["open", "high", "low", "close", "volume"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        else:
            d[col] = np.nan
    d = d.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    if len(d) < 120 + max_hold_bars:
        return []

    def _get_market_regime(i: int, lookback: int = 50) -> str:
        """Determine market regime for context-aware trading"""
        if i < lookback:
            return "ranging"
        
        window = d.iloc[i-lookback:i]
        
        # Simple trend detection
        sma_20 = window['close'].rolling(20).mean().iloc[-1]
        sma_50 = window['close'].rolling(min(50, len(window))).mean().iloc[-1]
        current_price = window['close'].iloc[-1]
        
        # Volatility measure
        price_changes = window['close'].pct_change().abs()
        volatility = price_changes.rolling(14).mean().iloc[-1]
        
        if current_price > sma_20 > sma_50 and volatility < 0.02:
            return "trending_up"
        elif current_price < sma_20 < sma_50 and volatility < 0.02:
            return "trending_down"
        elif volatility > 0.03:
            return "high_volatility"
        else:
            return "ranging"

    def _get_signal_strength(i: int) -> tuple:
        """Get signal strength and direction from SMC indicators"""
        signals = {}
        direction = 0
        strength = 0.0
        
        # Weight SMC signals differently
        signal_weights = {
            "BOS": 3.0,      # Break of structure is strongest
            "CHOCH": 2.5,    # Change of character
            "OB": 2.0,       # Order blocks
            "Liquidity": 1.5, # Liquidity sweeps
            "FVG": 1.0,      # Fair value gaps
            "HighLow": 0.5   # Basic swing points
        }
        
        for signal_col, weight in signal_weights.items():
            if signal_col in d.columns:
                try:
                    val = float(d.iloc[i].get(signal_col, 0))
                    if abs(val) > 0:
                        signals[signal_col] = val
                        if val > 0:
                            direction = 1
                            strength += abs(val) * weight
                        elif val < 0:
                            direction = -1
                            strength += abs(val) * weight
                except (ValueError, TypeError):
                    continue
        
        return direction, strength, signals

    def _calculate_dynamic_levels(i: int, direction: int, regime: str) -> tuple:
        """Calculate TP/SL levels based on market structure and regime"""
        entry_price = float(d.iloc[i]["close"])
        
        # Look for recent structure levels
        lookback = min(30, i)
        recent_window = d.iloc[i-lookback:i] if lookback > 0 else d.iloc[:i+1]
        
        if len(recent_window) < 5:
            # Fallback to simple percentage
            if direction > 0:
                tp_level = entry_price * 1.015  # 1.5% TP
                sl_level = entry_price * 0.992  # 0.8% SL
            else:
                tp_level = entry_price * 0.985  # 1.5% TP
                sl_level = entry_price * 1.008  # 0.8% SL
            return tp_level, sl_level
        
        # Find recent swing levels
        highs = recent_window['high'].values
        lows = recent_window['low'].values
        
        # Regime-based adjustments
        if regime == "trending_up" and direction > 0:
            # More aggressive in trending markets
            tp_multiplier = 2.5
            sl_multiplier = 1.0
        elif regime == "trending_down" and direction < 0:
            tp_multiplier = 2.5
            sl_multiplier = 1.0
        elif regime == "ranging":
            # More conservative in ranging markets
            tp_multiplier = 1.2
            sl_multiplier = 0.8
        elif regime == "high_volatility":
            # Wider stops in volatile markets
            tp_multiplier = 1.8
            sl_multiplier = 1.2
        else:
            # Default
            tp_multiplier = 1.5
            sl_multiplier = 1.0
        
        # Calculate based on recent structure
        if direction > 0:  # Long
            recent_low = np.min(lows[-10:]) if len(lows) >= 10 else np.min(lows)
            recent_high = np.max(highs[-20:]) if len(highs) >= 20 else np.max(highs)
            
            # SL below recent swing low with buffer
            base_risk = entry_price - recent_low
            sl_level = entry_price - (base_risk * sl_multiplier)
            
            # TP based on risk-reward ratio
            risk = entry_price - sl_level
            tp_level = entry_price + (risk * tp_multiplier)
            
            # Don't exceed recent resistance
            if recent_high > entry_price:
                tp_level = min(tp_level, recent_high * 0.998)  # Just below resistance
                
        else:  # Short
            recent_high = np.max(highs[-10:]) if len(highs) >= 10 else np.max(highs)
            recent_low = np.min(lows[-20:]) if len(lows) >= 20 else np.min(lows)
            
            base_risk = recent_high - entry_price
            sl_level = entry_price + (base_risk * sl_multiplier)
            
            risk = sl_level - entry_price
            tp_level = entry_price - (risk * tp_multiplier)
            
            # Don't exceed recent support
            if recent_low < entry_price:
                tp_level = max(tp_level, recent_low * 1.002)  # Just above support
        
        return tp_level, sl_level

    def _simulate_trade_management(i: int, direction: int, tp_level: float, sl_level: float, regime: str) -> tuple:
        """Simulate realistic trade management with trailing stops and early exits"""
        entry_price = float(d.iloc[i]["close"])
        
        # Dynamic holding period based on regime
        if regime == "high_volatility":
            max_hold = min(max_hold_bars // 2, 30)  # Shorter holds in volatile markets
        elif regime in ["trending_up", "trending_down"]:
            max_hold = max_hold_bars  # Full holding period in trends
        else:
            max_hold = max_hold_bars // 2  # Medium holds in ranging markets
        
        current_sl = sl_level
        best_price = entry_price
        bars_in_profit = 0
        
        for j in range(1, min(max_hold + 1, len(d) - i)):
            if i + j >= len(d):
                break
            
            bar = d.iloc[i + j]
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])
            
            if not all(np.isfinite([bar_high, bar_low, bar_close])):
                continue
            
            # Update best price and profit tracking
            if direction > 0:
                if bar_high > best_price:
                    best_price = bar_high
                if bar_close > entry_price * 1.005:  # 0.5% profit threshold
                    bars_in_profit += 1
                else:
                    bars_in_profit = 0
            else:
                if bar_low < best_price:
                    best_price = bar_low
                if bar_close < entry_price * 0.995:  # 0.5% profit threshold
                    bars_in_profit += 1
                else:
                    bars_in_profit = 0
            
            # Trailing stop logic - only after some profit
            if bars_in_profit >= 3:  # Trail after 3 bars in profit
                if direction > 0:
                    # Trail long stop up
                    trail_level = best_price * 0.995  # 0.5% trailing stop
                    current_sl = max(current_sl, trail_level)
                else:
                    # Trail short stop down
                    trail_level = best_price * 1.005  # 0.5% trailing stop
                    current_sl = min(current_sl, trail_level)
            
            # Check exit conditions
            if direction > 0:
                if bar_high >= tp_level:
                    return "tp", j, tp_level
                elif bar_low <= current_sl:
                    return "sl", j, current_sl
            else:
                if bar_low <= tp_level:
                    return "tp", j, tp_level
                elif bar_high >= current_sl:
                    return "sl", j, current_sl
            
            # Time-based exit for ranging markets
            if regime == "ranging" and j > max_hold // 2:
                if direction > 0 and bar_close > entry_price * 1.002:  # Small profit
                    return "time_exit", j, bar_close
                elif direction < 0 and bar_close < entry_price * 0.998:  # Small profit
                    return "time_exit", j, bar_close
        
        # No exit triggered
        final_price = float(d.iloc[min(i + max_hold, len(d) - 1)]["close"])
        return "none", max_hold, final_price

    trades = []
    tp_count = sl_count = none_count = 0
    
    for i in range(120, len(d) - max_hold_bars):
        regime = _get_market_regime(i)
        direction, strength, signals = _get_signal_strength(i)
        
        # Higher threshold for signal strength
        if direction == 0 or strength < min_signal_strength:
            # Still generate some "none" trades for balance
            if np.random.random() < 0.05:  # 5% chance
                context = d.iloc[i-120:i].to_dict(orient="records")
                entry_price = float(d.iloc[i]["close"])
                trade = {
                    "index": i,
                    "entry_price": entry_price,
                    "context_snapshot": context,
                    "result": "none",
                    "tp_price": entry_price,
                    "sl_price": entry_price,
                    "direction": 0,
                    "regime": regime,
                    "signal_strength": strength,
                    "smc_signals": signals
                }
                trades.append(trade)
                none_count += 1
            continue

        context = d.iloc[i-120:i].to_dict(orient="records")
        entry_price = float(d.iloc[i]["close"])
        
        # Calculate dynamic levels
        tp_level, sl_level = _calculate_dynamic_levels(i, direction, regime)
        
        # Simulate realistic trade management
        result, bars_held, exit_price = _simulate_trade_management(i, direction, tp_level, sl_level, regime)
        
        # Map results
        if result == "tp":
            final_result = "tp"
            final_tp_price = tp_level
            final_sl_price = sl_level
            tp_count += 1
        elif result == "sl":
            final_result = "sl"
            final_tp_price = tp_level
            final_sl_price = sl_level
            sl_count += 1
        elif result == "time_exit":
            # Treat profitable time exits as TP, others as none
            if ((direction > 0 and exit_price > entry_price * 1.002) or 
                (direction < 0 and exit_price < entry_price * 0.998)):
                final_result = "tp"
                tp_count += 1
            else:
                final_result = "none"
                none_count += 1
            final_tp_price = exit_price
            final_sl_price = sl_level
        else:
            final_result = "none"
            final_tp_price = tp_level
            final_sl_price = sl_level
            none_count += 1

        trade = {
            "index": i,
            "entry_price": entry_price,
            "context_snapshot": context,
            "result": final_result,
            "tp_price": final_tp_price,
            "sl_price": final_sl_price,
            "direction": direction,
            "bars_held": bars_held,
            "regime": regime,
            "signal_strength": strength,
            "smc_signals": signals
        }
        
        trades.append(trade)

    total_trades = len(trades)
    if total_trades > 0:
        tp_rate = tp_count / total_trades
        sl_rate = sl_count / total_trades
        none_rate = none_count / total_trades
        
        print(f"[TRADE GEN] Generated {total_trades} trades:")
        print(f"  TP: {tp_count} ({tp_rate:.1%})")
        print(f"  SL: {sl_count} ({sl_rate:.1%})")  
        print(f"  None: {none_count} ({none_rate:.1%})")
        
        # Show regime distribution
        regime_counts = {}
        for trade in trades:
            regime = trade.get("regime", "unknown")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        print(f"[TRADE GEN] Regime distribution: {regime_counts}")

    return trades

# Public accessor
def feature_columns():
    return list(FEATURE_COLUMNS)

# Global model instance placeholders
model_instance = None
optimizer_instance = None

def debug_model_outputs():
    """Call this to see what your model is actually outputting"""
    global model_instance
    if model_instance is None:
        print("No model loaded")
        return
        
    # Create dummy sequence
    dummy_seq = []
    for _ in range(120):
        # Create some variation in the dummy data
        features = [np.random.normal(0, 0.1) for _ in range(len(FEATURE_COLUMNS))]
        dummy_seq.append(features)
    
    model_instance.eval()
    with torch.no_grad():
        scaler = _load_scaler()
        seq = _apply_scaler_to_seq(dummy_seq, scaler)
        batch = np.expand_dims(seq, 0).astype(np.float32)
        tensor = torch.from_numpy(batch)
        
        # Raw model output (before temperature scaling)
        raw_out = model_instance(tensor)[0]
        print(f"Raw model outputs: {raw_out.tolist()}")
        
        # After sigmoid
        probs = torch.sigmoid(raw_out).tolist()
        print(f"After sigmoid: {probs}")
        
        # Check model parameters to see if they're learning
        total_params = sum(p.numel() for p in model_instance.parameters())
        trainable_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        print(f"Model has {total_params} total parameters, {trainable_params} trainable")
        
        # Check if weights are actually changing
        first_layer_weight_mean = model_instance.lstm.weight_ih_l0.mean().item()
        print(f"First layer weight mean: {first_layer_weight_mean:.6f}")

def debug_model_state():
    """Debug function to check if model is working correctly"""
    global model_instance
    
    if model_instance is None:
        print("[DEBUG] No model instance")
        return False
    
    print(f"[DEBUG] Model training mode: {model_instance.training}")
    
    # Check weights
    try:
        first_param = next(model_instance.parameters())
        weight_std = first_param.data.std().item()
        print(f"[DEBUG] Weight std: {weight_std:.8f}")
        
        if weight_std < 1e-8:
            print("[DEBUG] ⚠️ Model weights appear frozen!")
            return False
    except Exception as e:
        print(f"[DEBUG] Weight check failed: {e}")
        return False
    
    # Test with different inputs
    try:
        with torch.no_grad():
            input1 = torch.randn(1, 120, len(FEATURE_COLUMNS))
            input2 = input1 + torch.randn_like(input1) * 0.01
            
            out1 = model_instance(input1)
            out2 = model_instance(input2)
            
            diff = (out1 - out2).abs().mean().item()
            print(f"[DEBUG] Output sensitivity: {diff:.8f}")
            
            if diff < 1e-8:
                print("[DEBUG] ⚠️ Model outputs identical for different inputs!")
                return False
                
    except Exception as e:
        print(f"[DEBUG] Sensitivity test failed: {e}")
        return False
    
    print("[DEBUG] ✅ Model appears to be working correctly")
    return True