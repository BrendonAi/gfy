#imports
import os
import time
import traceback
import logging
try:
    import joblib
except Exception:
    joblib = None
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from typing import Dict, List, Optional, Tuple, Any
import warnings
import os
try:
    from trading_visualizer import TradingVisualizer
    import matplotlib.pyplot as plt
except ImportError:
    TradingVisualizer = None
    plt = None


# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

#tools
# --- SMC contract & utilities ---
REQUIRED_OHLC = ["open", "high", "low", "close", "volume"]
SCALER_PATH = "nn_feature_scaler.pkl"  # single source of truth for NN feature scaling

# Neural net feature schema must match the trained model exactly.
try:
    from nn_model import FEATURE_COLUMNS as _NN_FEATURE_COLUMNS
    FEATURE_COLUMNS_19 = list(_NN_FEATURE_COLUMNS)
except Exception:
    FEATURE_COLUMNS_19 = [
        # Basic OHLCV
        "open", "high", "low", "close", "volume",
        # SMC indicators from smc_logic.py
        "HighLow", "Level", "BOS", "CHOCH", "OB", "FVG", "Liquidity",
        # Derived features
        "HighLow_pos", "HighLow_neg",
        "TimeSinceBOS", "TimeSinceOB", "TimeSinceFVG", "TimeSinceLiquidity",
        "BarIndex"
    ]

# Hard assert to prevent silent schema drift
assert len(FEATURE_COLUMNS_19) == 19, f"Expected 19 features, got {len(FEATURE_COLUMNS_19)}: {FEATURE_COLUMNS_19}"

# --- Externalize trade fee ---
try:
    CONFIG
except NameError:
    CONFIG = {}
TRADE_FEE = CONFIG.get("trade_fee", 2.80)

# --- Runtime knobs (env-configurable) - Updated for full dataset ---
TAKE_THRESHOLD = float(os.getenv("TAKE_THRESHOLD", "0.60"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))
COOLDOWN_SEC   = int(float(os.getenv("COOLDOWN_SEC", "3")))
MAX_POS_SIZE   = int(os.getenv("MAX_POS_SIZE", "5"))
UNIT = int(os.getenv("BAR_UNIT", "2"))           # 2 == minute bars
UNIT_NUMBER = int(os.getenv("BAR_UNIT_NUMBER", "1"))
HIST_DAYS = int(os.getenv("HIST_DAYS", "30"))        # Changed from "3" to "30"
BAR_LIMIT = int(os.getenv("BAR_LIMIT", "30000"))      # Changed from "1500" to "30000"

# Ingestion/config knobs - Updated for full dataset
TICK_SIZE = float(os.getenv("TICK_SIZE", "0.25"))
STALE_TICK_SEC = float(os.getenv("STALE_TICK_SEC", "3"))
MAX_BARS_KEEP = int(os.getenv("MAX_BARS", "30000"))   # Changed from "2000" to "30000"
SESSION_MODE = os.getenv("SESSION", "24H").upper()
SESSION_UTC_START = os.getenv("SESSION_UTC_START", "13:30")
SESSION_UTC_END = os.getenv("SESSION_UTC_END", "20:00")

# Data quality thresholds
MAX_NAN_PCT = float(os.getenv("MAX_NAN_PCT", "0.1"))  # 10% max NaN per column
MIN_VARIATION_THRESHOLD = float(os.getenv("MIN_VARIATION", "1e-8"))
FEATURE_LOG_INTERVAL = int(os.getenv("FEATURE_LOG_INTERVAL", "100"))  # Log stats every N bars

# Set pandas options to avoid warnings
pd.set_option('future.no_silent_downcasting', False)

def _bucket_start(ts: pd.Timestamp, unit: int, unit_number: int) -> pd.Timestamp:
    """Calculate bucket start time for bar aggregation."""
    ts = pd.to_datetime(ts)
    if unit == 2:  # minute bars
        m = unit_number
        floored = ts - pd.to_timedelta(ts.minute % m, unit='m')
        return floored.replace(second=0, microsecond=0)
    return ts.replace(second=0, microsecond=0)

def _enforce_lower_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column case and ensure required OHLC columns exist."""
    d = df.copy()
    rename = {c: c.lower() for c in d.columns if c.lower() in {"timestamp","open","high","low","close","volume"}}
    if rename:
        d = d.rename(columns=rename)
    
    # Ensure required columns exist with proper defaults
    for c in REQUIRED_OHLC:
        if c not in d.columns:
            d[c] = 0 if c == "volume" else np.nan
    return d

def _time_since_non_nan(series: pd.Series) -> pd.Series:
    """Calculate time since last non-NaN value in series."""
    idx = series.index
    out = np.zeros(len(series), dtype=float)
    last = -1
    
    for i, v in enumerate(series.fillna(np.nan)):
        if not pd.isna(v):
            last = i
            out[i] = 0.0
        else:
            out[i] = (i - last) if last >= 0 else float(i)
    
    return pd.Series(out, index=idx)

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    try:
        x = float(x)
        if abs(x) > 500:
            return 1.0 if x > 0 else 0.0
    except Exception:
        return 0.5
    
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

# --- UTC and session helpers ---
def _to_utc_naive(ts) -> pd.Timestamp:
    """Convert timestamp to UTC naive for consistent datetime math."""
    t = pd.to_datetime(ts, utc=True)
    return t.tz_convert('UTC').tz_localize(None)

def _in_session_utc(ts: pd.Timestamp) -> bool:
    """Check if timestamp falls within trading session."""
    if SESSION_MODE == '24H':
        return True
    
    try:
        h_start, m_start = map(int, SESSION_UTC_START.split(':'))
        h_end, m_end = map(int, SESSION_UTC_END.split(':'))
    except Exception:
        return True
    
    start = ts.replace(hour=h_start, minute=m_start, second=0, microsecond=0)
    end = ts.replace(hour=h_end, minute=m_end, second=0, microsecond=0)
    
    if end >= start:
        return start <= ts <= end
    return ts >= start or ts <= end

# --- ROBUST FEATURE ENGINEERING ---
class FeatureValidator:
    """Validates and monitors feature quality for neural network input."""
    
    def __init__(self):
        self.feature_stats_history = []
        self.bar_count = 0
        
    def validate_sequence_quality(self, seq_120x19: List[List[float]], context: str = "") -> bool:
        """Comprehensive validation of 120x19 sequence before NN inference."""
        try:
            arr = np.array(seq_120x19, dtype=np.float32)
            
            # Shape validation
            expected_shape = (120, 19)
            if arr.shape != expected_shape:
                logging.error(f"[VALIDATION] {context} - Wrong shape: {arr.shape}, expected {expected_shape}")
                return False
            
            # Check for invalid values
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                logging.error(f"[VALIDATION] {context} - Invalid values detected")
                return False
            
            # Check for reasonable value ranges and variation
            if np.any(np.abs(arr) > 1e6):
                logging.warning(f"[VALIDATION] {context} - Extreme values detected")
            
            if np.std(arr) < MIN_VARIATION_THRESHOLD:
                logging.warning(f"[VALIDATION] {context} - Insufficient variation")
            
            return True
            
        except Exception as e:
            logging.error(f"[VALIDATION] {context} - Validation failed: {e}")
            return False
    
    def log_feature_stats(self, smc_df: pd.DataFrame, context: str = ""):
        """Log comprehensive feature statistics for monitoring."""
        self.bar_count += 1
        
        if self.bar_count % FEATURE_LOG_INTERVAL != 0:
            return
            
        try:
            if len(smc_df) >= 120:
                tail = smc_df.tail(120)
                
                # Check for data quality issues
                nan_issues = []
                constant_issues = []
                
                for col in FEATURE_COLUMNS_19:
                    if col in tail.columns:
                        data = tail[col]
                        if data.isnull().sum() > 12:  # >10% NaN
                            nan_issues.append(col)
                        if data.std() < MIN_VARIATION_THRESHOLD:
                            constant_issues.append(col)
                
                if nan_issues or constant_issues:
                    logging.warning(f"[FEATURE_STATS] {context} - Issues: NaN={nan_issues}, Constant={constant_issues}")
                
        except Exception as e:
            logging.error(f"[FEATURE_STATS] Failed to log stats: {e}")

def _ensure_feature_schema_robust(df: pd.DataFrame) -> pd.DataFrame:
    """Robust feature schema enforcement. Returns EXACTLY the 19 features in correct order."""
    df = df.copy()
    
    # Handle SMC indicators with proper forward-fill
    smc_indicators = ["HighLow", "Level", "BOS", "CHOCH", "OB", "FVG", "Liquidity"]
    for indicator in smc_indicators:
        if indicator not in df.columns:
            df[indicator] = np.nan
        elif not df[indicator].empty:
            df[indicator] = df[indicator].ffill()
    
    # Handle time-since features
    time_features = ["TimeSinceBOS", "TimeSinceOB", "TimeSinceFVG", "TimeSinceLiquidity"]
    for time_feat in time_features:
        if time_feat not in df.columns:
            df[time_feat] = np.arange(len(df), dtype=float)
        elif not df[time_feat].empty:
            df[time_feat] = df[time_feat].ffill().fillna(0)
    
    # Handle BarIndex
    if "BarIndex" not in df.columns:
        df["BarIndex"] = np.arange(len(df), dtype=float)
    
    # Split HighLow into pos/neg
    df["HighLow_pos"] = (df.get("HighLow", 0) == 1).astype(float)
    df["HighLow_neg"] = (df.get("HighLow", 0) == -1).astype(float)
    
    # Ensure OHLCV exists
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            if c == "volume":
                df[c] = 0.0
            else:
                close_val = df.get("close", pd.Series([100.0] * len(df)))
                df[c] = close_val.iloc[-1] if hasattr(close_val, 'iloc') and len(close_val) > 0 and not pd.isna(close_val.iloc[-1]) else 100.0
    
    # Replace infinities and handle NaNs
    df = df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
    
    # Build result with exactly 19 features in correct order
    result_df = pd.DataFrame(index=df.index)
    for c in FEATURE_COLUMNS_19:
        if c in df.columns:
            series = df[c].copy()
            if c in ["open", "high", "low", "close"]:
                series = series.ffill().bfill()
            result_df[c] = series.fillna(0.0)
        else:
            # Create missing column with appropriate default
            if c == "volume":
                result_df[c] = 0.0
            elif c in ["open", "high", "low", "close"]:
                close_val = df.get("close", pd.Series([100.0] * len(df)))
                result_df[c] = close_val.iloc[-1] if hasattr(close_val, 'iloc') and len(close_val) > 0 and not pd.isna(close_val.iloc[-1]) else 100.0
            else:
                result_df[c] = 0.0
    
    return result_df

def _build_validated_sequence(df_tail_120: pd.DataFrame, validator: FeatureValidator) -> Optional[List[List[float]]]:
    """Build sequence with comprehensive validation."""
    
    if len(df_tail_120) != 120:
        logging.error(f"Expected 120 bars, got {len(df_tail_120)}")
        return None
    
    # Check for required columns
    missing_cols = set(FEATURE_COLUMNS_19) - set(df_tail_120.columns)
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        return None
    
    try:
        feature_data = df_tail_120[FEATURE_COLUMNS_19].copy()
        
        # Check data quality
        nan_pct = feature_data.isnull().sum() / len(feature_data)
        problematic_cols = nan_pct[nan_pct > MAX_NAN_PCT].index.tolist()
        if problematic_cols:
            logging.error(f"High NaN percentage in columns: {problematic_cols}")
            return None
        
        # Build matrix
        mat = feature_data.astype(np.float32).values
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        
        if mat.shape != (120, 19):
            logging.error(f"Matrix shape mismatch: {mat.shape}, expected (120, 19)")
            return None
        
        # Validate sequence
        sequence = mat.tolist()
        return sequence if validator.validate_sequence_quality(sequence, "sequence_build") else None
        
    except Exception as e:
        logging.error(f"Failed to build sequence: {e}")
        return None

# --- Import external dependencies ---
from signalr import (
    login_and_get_token,
    SignalRClient,
    place_order,
    fetch_bars,
    get_latest_contract_id,
    search_open_orders,
    cancel_order,
    get_equity_and_pnl,
    get_active_accounts,
)

from smc_logic import smc
from nn_model import online_learn_from_trade, predict_sequence, extract_features_from_bar, _ensure_smc_context

# --- SIMPLE MARKET LEARNING ---
def create_market_learning_data(df, lookforward=15):
    """Simple market prediction approach."""
    
    if len(df) < 120 + lookforward:
        return [], []
    
    df_with_smc = _ensure_smc_context(df)
    sequences = []
    labels = []
    
    print(f"[SIMPLE] Processing {len(df_with_smc)} bars with {lookforward}-bar lookforward...")
    
    for i in range(120, len(df_with_smc) - lookforward):
        context_bars = df_with_smc.iloc[i-120:i]
        current_price = float(df_with_smc.iloc[i]['close'])
        
        # Look forward and see what happened
        future_bars = df_with_smc.iloc[i+1:i+1+lookforward]
        future_highs = future_bars['high'].values
        future_lows = future_bars['low'].values
        future_closes = future_bars['close'].values
        
        # Calculate movements
        max_gain = (future_highs.max() - current_price) / current_price
        max_loss = (current_price - future_lows.min()) / current_price  
        final_return = (future_closes[-1] - current_price) / current_price
        
        # Simple labeling logic
        significant_move = max(abs(max_gain), abs(max_loss)) > 0.008
        take_trade = 1.0 if significant_move else 0.0
        
        move_strength = max(abs(max_gain), abs(max_loss))
        confidence = min(1.0, move_strength / 0.02)
        
        if max_gain > abs(max_loss):
            tp_offset = max_gain
            sl_offset = max_loss
            direction_long = True
        else:
            tp_offset = max_loss
            sl_offset = max_gain
            direction_long = False
            
        # Quality metrics
        if direction_long:
            tp_quality = 1.0 if final_return > 0.01 else 0.0
            sl_quality = 1.0 if final_return < -0.01 else 0.0
        else:
            tp_quality = 1.0 if final_return < -0.01 else 0.0
            sl_quality = 1.0 if final_return > 0.01 else 0.0
            
        # Build feature sequence
        sequence = [extract_features_from_bar(row) for _, row in context_bars.iterrows()]
            
        if len(sequence) == 120:
            sequences.append(sequence)
            label = [
                float(take_trade),
                float(confidence), 
                float(min(0.05, max(-0.05, tp_offset))),
                float(min(0.05, max(-0.05, sl_offset))),
                float(tp_quality),
                float(sl_quality)
            ]
            labels.append(label)
    
    print(f"[SIMPLE] Generated {len(sequences)} learning examples")
    
    if sequences:
        labels_array = np.array(labels)
        take_rate = (labels_array[:, 0] > 0.5).mean()
        print(f"[SIMPLE] Take rate: {take_rate:.3f}")
    
    return sequences, labels

def create_every_bar_learning_data(df, lookforward=3):
    
    if len(df) < 120 + lookforward:
        return [], []
        
    df_with_smc = _ensure_smc_context(df)
    sequences = []
    labels = []
    
    print(f"[ULTRA_SIMPLE] Processing every bar with {lookforward}-bar lookforward...")
    
    for i in range(120, len(df_with_smc) - lookforward):
        context = df_with_smc.iloc[i-120:i]
        current = float(df_with_smc.iloc[i]['close'])
        
        next_bars = df_with_smc.iloc[i+1:i+1+lookforward]
        next_high = next_bars['high'].max()
        next_low = next_bars['low'].min()
        next_close = next_bars['close'].iloc[-1]
        
        up_move = (next_high - current) / current
        down_move = (current - next_low) / current
        final_move = (next_close - current) / current
        
        should_trade = 1.0 if abs(final_move) > 0.002 else 0.0
        confidence = min(1.0, abs(final_move) / 0.01)
        
        sequence = [extract_features_from_bar(row) for _, row in context.iterrows()]
        
        if len(sequence) == 120:
            sequences.append(sequence)
            labels.append([
                should_trade,
                confidence,
                up_move,
                down_move, 
                1.0 if final_move > 0.002 else 0.0,
                1.0 if final_move < -0.002 else 0.0
            ])
    
    print(f"[ULTRA_SIMPLE] Generated {len(sequences)} learning examples")
    return sequences, labels

def train_simple_market_model(historical_df, approach="realistic", **train_kwargs):
    """Fixed training pipeline with realistic trade generation."""
    from nn_model import train_model, convert_to_trades, prepare_data
    
    print("Realistic Market Learning: Using human-like trade management!")
    
    # Use the improved convert_to_trades function
    trades = convert_to_trades(historical_df, max_hold_bars=45, min_signal_strength=2.0)
    
    if not trades:
        print("[ERROR] No trades generated!")
        return None
        
    X_sequences, y_labels = prepare_data(trades)
    
    if not X_sequences:
        print("[ERROR] No training sequences generated!")
        return None
        
    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32)
    
    print(f"[TRAIN] Training on {len(X)} sequences with shape {X.shape}")
    
    input_size = len(FEATURE_COLUMNS_19)
    model = train_model(X, y, input_size, **train_kwargs)
    
    return model, extract_features_from_bar, _ensure_smc_context

# --- UNIFIED DATA PIPELINE ---
class DataPipeline:
    """Unified data processing pipeline for both sim and live modes."""
    
    def __init__(self, mode: str = 'live'):
        self.mode = mode
        self.validator = FeatureValidator()
        self._last_smc_ts = None
        self._last_smc_df = None
        self._smc_computation_count = 0
        
    def standardize_tick_data(self, tick_data: Dict, is_live: bool = True) -> Optional[Dict]:
        """Standardize tick data format between live and sim modes."""
        
        price = tick_data.get("price") or tick_data.get("p")
        ts_raw = tick_data.get("timestamp") or tick_data.get("time")
        volume = tick_data.get("volume") or tick_data.get("q") or 0
        
        if price is None or ts_raw is None:
            return None
        
        try:
            ts = _to_utc_naive(ts_raw)
        except Exception:
            logging.warning(f"Invalid timestamp in tick: {ts_raw}")
            return None
        
        if is_live and (datetime.utcnow() - ts).total_seconds() > STALE_TICK_SEC:
            return None
        
        if not _in_session_utc(ts):
            return None
        
        try:
            price = float(price)
            ticks = round(price / TICK_SIZE)
            price = ticks * TICK_SIZE
            volume = max(0, int(volume))
        except Exception:
            return None
        
        return {
            "timestamp": ts,
            "price": price,
            "volume": volume
        }
    
    def get_consistent_smc_features(self, aggregator, force_recompute: bool = False) -> pd.DataFrame:
        df_now = aggregator.get_df()
        if df_now.empty:
            raise ValueError("Aggregator DataFrame is empty")
        
        df_now = _enforce_lower_ohlc(df_now)
        last_bar_ts = df_now.iloc[-1]["timestamp"]
        last_bar_close = df_now.iloc[-1]["close"]
        
        # Enhanced change detection
        should_recompute = (
            force_recompute or 
            self._last_smc_ts is None or 
            self._last_smc_ts != last_bar_ts or
            len(df_now) != getattr(self, '_last_df_len', 0) or
            abs(last_bar_close - getattr(self, '_last_close', 0)) > 1e-6
        )
        
        print(f"[SMC DEBUG] should_recompute: {should_recompute}, last_ts: {self._last_smc_ts}, current_ts: {last_bar_ts}")
        
        if should_recompute:
            df_for_smc = df_now.tail(5000) if len(df_now) > 5000 else df_now
            smc_df = self._compute_smc_features(df_for_smc)
            
            self._last_smc_ts = last_bar_ts
            self._last_df_len = len(df_now)
            self._last_close = last_bar_close  # Track price changes
            self._last_smc_df = smc_df.copy()
            self._smc_computation_count += 1
            
            print(f"[SMC] RECOMPUTED features, computation #{self._smc_computation_count}")
            return smc_df
        else:
            print(f"[SMC] USING CACHED features")
            return self._last_smc_df.copy() if self._last_smc_df is not None else self._compute_smc_features(df_now.tail(5000) if len(df_now) > 5000 else df_now)
            
    def _compute_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute SMC features with error handling."""
        def safe_smc_compute(func, *args, **kwargs):
            """Helper to safely compute SMC components."""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"[SMC] {func.__name__} computation failed: {e}")
                return pd.DataFrame(index=df.index)
        
        try:
            ohlc = df[REQUIRED_OHLC].copy()
            logging.debug(f"[SMC] Computing features on {len(ohlc)} bars")
            
            # Compute all SMC components
            sw = safe_smc_compute(smc.swing_highs_lows, ohlc, swing_length=5).add_prefix("SW_")
            if sw.empty:
                sw = pd.DataFrame({"SW_HighLow": np.nan, "SW_Level": np.nan}, index=ohlc.index)
            
            swings_renamed = sw.rename(columns={"SW_HighLow": "HighLow", "SW_Level": "Level"})
            ms = safe_smc_compute(smc.bos_choch, ohlc, swings_renamed).add_prefix("MS_")
            obdf = safe_smc_compute(smc.ob, ohlc, swings_renamed, close_mitigation=True).add_prefix("OB_")
            fvgd = safe_smc_compute(smc.fvg, ohlc).add_prefix("FVG_")
            liqd = safe_smc_compute(smc.liquidity, ohlc, swings_renamed, range_percent=0.005).add_prefix("LIQ_")
            
            # Combine all features
            smc_df = pd.concat([df.reset_index(drop=True), sw, ms, obdf, fvgd, liqd], axis=1)
            smc_df = smc_df.loc[:, ~smc_df.columns.duplicated()]
            
            # Calculate time-since features
            for col, prefix in [("MS_BOS", "TimeSinceBOS"), ("OB_OB", "TimeSinceOB"), 
                              ("FVG_FVG", "TimeSinceFVG"), ("LIQ_Liquidity", "TimeSinceLiquidity")]:
                smc_df[prefix] = _time_since_non_nan(smc_df.get(col, pd.Series(np.nan, index=smc_df.index)))
            
            smc_df["BarIndex"] = np.arange(len(smc_df), dtype=float)
            
            # Map main SMC columns
            smc_df["HighLow"] = smc_df.get("SW_HighLow", np.nan)
            smc_df["OB"] = smc_df.get("OB_OB", np.nan)
            smc_df["FVG"] = smc_df.get("FVG_FVG", np.nan)
            smc_df["Liquidity"] = smc_df.get("LIQ_Liquidity", np.nan)
            
            # Apply robust feature schema
            smc_df = _ensure_feature_schema_robust(smc_df)
            
            # Log stats periodically
            self.validator.log_feature_stats(smc_df, "smc_computation")
            
            return smc_df
            
        except Exception as e:
            logging.error(f"SMC computation failed: {e}")
            return _ensure_feature_schema_robust(df)

# --- DATA AGGREGATOR ---
class BarAggregator:
    """Enhanced OHLCV bar aggregator with validation."""
    
    FLUSH_THRESHOLD = 100
    
    def __init__(self):
        self._df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self._bars = []
        self._buffered_bars = []
        self._max_bars = MAX_BARS_KEEP
        self._last_bucket_ts = None
        self._tick_count = 0
        
    def _clean_ohlc(self, bar: dict) -> dict:
        """Clean and validate OHLC bar data."""
        ohlc = {}
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            value = bar.get(col)
            ohlc[col] = 0 if col == "volume" and value is None else (value if value is not None else np.nan)
        return ohlc
    
    def add_bar(self, bar: dict):
        """Add a complete bar to the aggregator."""
        ohlc = self._clean_ohlc(bar)
        self._buffered_bars.append([ohlc[col] for col in ["timestamp", "open", "high", "low", "close", "volume"]])
        
        if len(self._buffered_bars) >= self.FLUSH_THRESHOLD:
            self.flush()
    
    def flush(self):
        """Flush buffered bars to main DataFrame."""
        if len(self._buffered_bars) >= self.FLUSH_THRESHOLD:
            new_df = pd.DataFrame(self._buffered_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            self._df = pd.concat([self._df, new_df], ignore_index=True)
            self._buffered_bars = []
            
            if not self._df.empty:
                self._bars = self._df.to_dict('records')
    
    def add_tick(self, tick: dict) -> bool:
        """Add tick data and return True if a new bar was created."""
        
        price = tick.get("price")
        ts = tick.get("timestamp")
        volume = tick.get("volume", 0)
        
        if price is None or ts is None:
            return False
        
        self._tick_count += 1
        
        try:
            price = float(price)
            volume = max(0, int(volume))
        except Exception:
            return False
        
        bucket_ts = _bucket_start(ts, UNIT, UNIT_NUMBER)
        
        # DEBUG: Check for timestamp regression
        if self._last_bucket_ts is not None and bucket_ts < self._last_bucket_ts:
            if self._tick_count % 100 == 0:  # Don't spam this warning
                print(f"[AGGREGATOR] WARNING: Timestamp regression detected: {bucket_ts} < {self._last_bucket_ts}")
            return False
        
        new_bar = False
        
        # DEBUG: Track previous state for comparison
        prev_bar_count = len(self._df)
        prev_close = self._df.iloc[-1]["close"] if not self._df.empty else None
        
        if self._df.empty:
            self._df = pd.concat([self._df, pd.DataFrame([{
                "timestamp": bucket_ts, "open": price, "high": price,
                "low": price, "close": price, "volume": volume
            }])], ignore_index=True)
            new_bar = True
            print(f"[AGGREGATOR] FIRST BAR: {bucket_ts}, price: {price}")
            
        else:
            last_row = self._df.iloc[-1]
            last_ts = last_row["timestamp"]
            
            if bucket_ts != last_ts:
                self._df = pd.concat([self._df, pd.DataFrame([{
                    "timestamp": bucket_ts, "open": price, "high": price,
                    "low": price, "close": price, "volume": volume
                }])], ignore_index=True)
                new_bar = True
                
                # DEBUG: Log new bar creation with details
                time_diff = (bucket_ts - last_ts).total_seconds() if hasattr(bucket_ts - last_ts, 'total_seconds') else 'unknown'
                print(f"[AGGREGATOR] NEW BAR: {bucket_ts} (prev: {last_ts}, gap: {time_diff}s)")
                print(f"   Price: {prev_close} -> {price}, Bars: {prev_bar_count} -> {len(self._df)}")
                
            else:
                # Update existing bar
                i = self._df.index[-1]
                old_high = self._df.at[i, "high"]
                old_low = self._df.at[i, "low"]
                old_close = self._df.at[i, "close"]
                old_volume = self._df.at[i, "volume"]
                
                self._df.at[i, "high"] = max(old_high, price)
                self._df.at[i, "low"] = min(old_low, price)
                self._df.at[i, "close"] = price
                self._df.at[i, "volume"] = old_volume + volume
                
                # DEBUG: Log bar updates, but not too frequently
                if self._tick_count % 50 == 0 or abs(price - old_close) > 0.5:  # Log every 50 ticks or significant price move
                    print(f"[AGGREGATOR] UPDATE BAR: price {old_close} -> {price}, "
                        f"H: {old_high} -> {self._df.at[i, 'high']}, "
                        f"L: {old_low} -> {self._df.at[i, 'low']}, "
                        f"V: {old_volume} -> {self._df.at[i, 'volume']} (tick #{self._tick_count})")
        
        # Maintain bounded history
        if len(self._df) > self._max_bars:
            removed_bars = len(self._df) - self._max_bars
            self._df = self._df.iloc[-self._max_bars:].reset_index(drop=True)
            if removed_bars > 0:
                print(f"[AGGREGATOR] TRIMMED {removed_bars} old bars, kept {len(self._df)} bars")
        
        # Update records
        self._bars = self._df.to_dict('records')
        
        if not self._df.empty:
            self._last_bucket_ts = self._df.iloc[-1]["timestamp"]
        
        # DEBUG: Additional validation every 100 ticks
        if self._tick_count % 100 == 0:
            latest_bar = self._df.iloc[-1]
            print(f"[AGGREGATOR] STATUS: {self._tick_count} ticks processed, {len(self._df)} bars")
            print(f"   Latest bar: {latest_bar['timestamp']} | OHLCV: "
                f"{latest_bar['open']:.2f}/{latest_bar['high']:.2f}/"
                f"{latest_bar['low']:.2f}/{latest_bar['close']:.2f}/{latest_bar['volume']}")
            
            # Check for data quality issues
            if latest_bar['high'] < latest_bar['low']:
                print(f"[AGGREGATOR] ERROR: Invalid OHLC - High < Low!")
            if latest_bar['open'] > latest_bar['high'] or latest_bar['open'] < latest_bar['low']:
                print(f"[AGGREGATOR] ERROR: Invalid OHLC - Open outside H/L range!")
            if latest_bar['close'] > latest_bar['high'] or latest_bar['close'] < latest_bar['low']:
                print(f"[AGGREGATOR] ERROR: Invalid OHLC - Close outside H/L range!")
        
        # DEBUG: Check for potential stuck data
        if not new_bar and hasattr(self, '_last_update_price'):
            if self._last_update_price == price and self._tick_count > 10:
                if not hasattr(self, '_stuck_price_count'):
                    self._stuck_price_count = 0
                self._stuck_price_count += 1
                
                if self._stuck_price_count > 20:  # Same price for 20+ ticks
                    print(f"[AGGREGATOR] WARNING: Price stuck at {price} for {self._stuck_price_count} ticks")
                    self._stuck_price_count = 0  # Reset counter
            else:
                self._stuck_price_count = 0
        
        self._last_update_price = price
        
        return new_bar
    
    def replace_bars(self, bar_list):
        """Replace internal DataFrame with provided bars."""
        if not bar_list:
            self._df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            self._bars = []
            return
        
        df = pd.DataFrame(bar_list)
        df = df.rename(columns={c: c.lower() for c in df.columns})
        
        # Ensure required columns
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                df[col] = np.nan
        if "volume" not in df.columns:
            df["volume"] = 0
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        self._df = df
        
        if len(self._df) > self._max_bars:
            self._df = self._df.iloc[-self._max_bars:].reset_index(drop=True)
        
        self._bars = self._df.to_dict('records')
    
    def get_df(self) -> pd.DataFrame:
        """Get copy of internal DataFrame."""
        return self._df.copy(deep=False)
    
    def is_ready(self) -> bool:
        """Check if aggregator has data."""
        return not self._df.empty and len(self._df) >= 1
    
    def latest_bar(self) -> Optional[Dict]:
        """Get the latest bar."""
        return self._df.iloc[-1].to_dict() if not self._df.empty else None

# --- ENHANCED PROCESSOR ---
class Processor:
    """Enhanced processor with robust data handling and validation."""
    
    def __init__(self, account_id, contract_id, token, mode='live'):
        self.account_id = account_id
        self.contract_id = contract_id
        self.token = token
        self.mode = mode
        
        # Trading state
        self.open_trades = {}
        self._current_price = None
        self._last_trade_close_time = 0
        self._position_size = 0
        self._entry_price = None
        self._entry_order_id = None
        
        # Data pipeline
        self.data_pipeline = DataPipeline(mode)
        
        # Callbacks and metrics
        self.metrics_cb = None
        
        # Feature scaling
        self._scaler = None
        if joblib is not None:
            try:
                self._scaler = joblib.load(SCALER_PATH)
                logging.info(f"[NN] Loaded feature scaler: {SCALER_PATH}")
            except Exception:
                logging.warning("[NN] No scaler found, proceeding without scaling")
        
        # Thresholds
        self.take_threshold = TAKE_THRESHOLD
        self.conf_threshold = CONF_THRESHOLD
        self.cooldown_sec = COOLDOWN_SEC
        self.max_position_size = MAX_POS_SIZE
        
        # Tick filtering
        self._last_tick_time = None
        self._processed_tick_count = 0

        self.visualizer = None
        if TradingVisualizer and os.getenv("ENABLE_VIZ", "false").lower() == "true":
            self.visualizer = TradingVisualizer()
            print("[VIZ] Visualization enabled")
    
    def _apply_robust_scaling(self, seq_120x19: List[List[float]]) -> List[List[float]]:
        """Apply scaling with comprehensive validation."""
        if self._scaler is None:
            return seq_120x19
        
        try:
            arr = np.asarray(seq_120x19, dtype=np.float32)
            expected_features = getattr(self._scaler, 'n_features_in_', len(FEATURE_COLUMNS_19))
            
            if arr.shape[-1] != expected_features:
                logging.warning(f"Feature count mismatch: {arr.shape[-1]} vs {expected_features}")
                return seq_120x19
            
            if np.any(np.isinf(arr)) or np.any(np.isnan(arr)):
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            
            orig_shape = arr.shape
            arr2d = arr.reshape(-1, orig_shape[-1])
            scaled = self._scaler.transform(arr2d)
            scaled = scaled.reshape(orig_shape)
            
            if np.any(np.isinf(scaled)) or np.any(np.isnan(scaled)):
                logging.warning("Scaling produced invalid values")
                return seq_120x19
            
            return scaled.astype(np.float32).tolist()
            
        except Exception as e:
            logging.error(f"Scaling failed: {e}")
            return seq_120x19
    
    def process_tick(self, tick, aggregator):
        """Main tick processing with unified data pipeline."""
        
        std_tick = self.data_pipeline.standardize_tick_data(tick, is_live=(self.mode == 'live'))
        if std_tick is None:
            return None
        
        if self._last_tick_time == std_tick["timestamp"]:
            return None
        self._last_tick_time = std_tick["timestamp"]
        
        self._processed_tick_count += 1
        self._current_price = std_tick["price"]
        
        new_bar = aggregator.add_tick(std_tick)
        
        if not aggregator.is_ready():
            return None
        
        df_now = aggregator.get_df()
        if df_now.empty:
            return None
        
        if self.mode == 'live':
            self.check_protective_orders_status()
        
        try:
            smc_df = self.data_pipeline.get_consistent_smc_features(aggregator, force_recompute=new_bar)
        except Exception as e:
            logging.error(f"Failed to get SMC features: {e}")
            return None
        
        if len(smc_df) < 120:
            return None
        
        tail_df = smc_df.tail(120)
        sequence = _build_validated_sequence(tail_df, self.data_pipeline.validator)
        
        if sequence is None or len(sequence) != 120 or len(sequence[0]) != len(FEATURE_COLUMNS_19):
            return None
        
        scaled_sequence = self._apply_robust_scaling(sequence)
        
        try:
            prediction = predict_sequence(scaled_sequence)
            pred_array = np.array(prediction).flatten()
            if len(pred_array) < 6:
                logging.error(f"Model returned {len(pred_array)} outputs, expected 6")
                return None
                
            take_logit, conf_logit, tp_offset, sl_offset, tp_quality_logit, sl_quality_logit = map(float, pred_array[:6])
        except Exception as e:
            logging.error(f"NN prediction failed: {e}")
            return None
        
        p_take = _sigmoid(take_logit)
        p_conf = _sigmoid(conf_logit)
        is_long_pred = (tp_offset - sl_offset) > 0
        
        if new_bar or self._processed_tick_count % 100 == 0:
            logging.info(f"[NN PRED] p_take={p_take:.3f}, p_conf={p_conf:.3f}, "
                        f"tp_offset={tp_offset:.5f}, sl_offset={sl_offset:.5f}, "
                        f"dir={'LONG' if is_long_pred else 'SHORT'}")
        
        if self.metrics_cb:
            try:
                self.metrics_cb({
                    "timestamp": std_tick["timestamp"],
                    "p_take": p_take,
                    "p_conf": p_conf,
                    "direction": "LONG" if is_long_pred else "SHORT",
                    "price": self._current_price,
                })
            except Exception as e:
                logging.debug(f"Metrics callback failed: {e}")
        
        if self.mode == 'live':
            return self._execute_trading_logic(p_take, p_conf, is_long_pred, tp_offset, sl_offset, smc_df)
        
        if self.visualizer and new_bar:
            try:
                latest_bar = aggregator.latest_bar()
                smc_latest = smc_df.iloc[-1].to_dict() if not smc_df.empty else {}
                
                pred_dict = {
                    "p_take": p_take,
                    "p_conf": p_conf,
                    "direction": "LONG" if is_long_pred else "SHORT", 
                    "tp_offset": tp_offset,
                    "sl_offset": sl_offset
                }
                
                # Add data point
                self.visualizer.add_data_point(latest_bar, smc_latest, pred_dict, sequence[-1] if sequence else [])
                
                # Update plots every 10 bars to avoid lag
                if self._processed_tick_count % 10 == 0:
                    self.visualizer.update_plots()
                    if plt:
                        plt.pause(0.01)
                        
                # Debug features every 50 bars  
                if self._processed_tick_count % 50 == 0:
                    self.visualizer.show_feature_debug()
                    
            except Exception as e:
                logging.debug(f"Visualization failed: {e}")

        return "PROCESSED"
    
    def _execute_trading_logic(self, p_take, p_conf, is_long_pred, tp_offset, sl_offset, smc_df):
        """Execute trading logic with proper validation."""
        
        current_time = time.time()
        should_trade = (p_take >= self.take_threshold) and (p_conf >= self.conf_threshold)
        safe_to_trade = (
            should_trade and 
            not self.open_trades and 
            (current_time - self._last_trade_close_time) > self.cooldown_sec
        )
        
        if safe_to_trade and self._position_size == 0:
            try:
                account_info = get_equity_and_pnl(token=self.token, account_id=self.account_id)
                account_balance = account_info.get("balance", account_info.get("equity", 0))
                desired_size = self.get_lot_size(account_balance)
                
                if desired_size >= 1:
                    order_id = self._place_entry_order(is_long_pred, desired_size)
                    if order_id:
                        self._position_size = desired_size
                        self._entry_price = self._current_price
                        self._entry_order_id = order_id
                        
                        logging.info(f"[TRADE] Opened {'LONG' if is_long_pred else 'SHORT'} "
                                   f"position: {desired_size} contracts at {self._entry_price}")
                        
            except Exception as e:
                logging.error(f"Trading execution failed: {e}")
        
        if (self._entry_order_id and 
            self._entry_order_id not in self.open_trades and 
            self._entry_price and self._position_size >= 1):
            
            try:
                self.place_protective_orders(
                    self._position_size, self._entry_price, is_long_pred,
                    sl_offset, tp_offset, smc_df.iloc[-1],
                    self._entry_order_id, self._entry_order_id
                )
            except Exception as e:
                logging.error(f"Protective order placement failed: {e}")
        
        return "TRADE_PROCESSED"
    
    def _place_entry_order(self, is_long: bool, size: int) -> Optional[str]:
        """Place entry order with retry logic."""
        order_params = {
            "account_id": self.account_id,
            "contract_id": self.contract_id,
            "order_type": 2,  # Market order
            "side": 0 if is_long else 1,
            "size": size
        }
        
        for attempt in range(3):
            try:
                order_id = place_order(token=self.token, **order_params)
                return order_id
            except Exception as e:
                logging.error(f"Entry order attempt {attempt+1} failed: {e}")
                time.sleep(1)
        
        return None
    
    def get_lot_size(self, account_balance: float) -> int:
        """Calculate position size based on account balance."""
        if account_balance < 1500:
            return 2
        elif 1500 <= account_balance <= 2000:
            return 3
        else:
            return 5
    
    def check_protective_orders_status(self):
        """Check and manage protective orders."""
        try:
            open_orders = search_open_orders(token=self.token, account_id=self.account_id)
            open_order_ids = {order.get("orderId") for order in open_orders}
            
            trades_to_remove = []
            for trade_id, trade in self.open_trades.items():
                sl_order_id = trade.get("sl_order_id")
                tp_order_id = trade.get("tp_order_id")
                
                sl_filled = sl_order_id and sl_order_id not in open_order_ids
                tp_filled = tp_order_id and tp_order_id not in open_order_ids
                
                if sl_filled or tp_filled:
                    logging.info(f"Trade {trade_id} closed via {'SL' if sl_filled else 'TP'}")
                    self._last_trade_close_time = time.time()
                    self._position_size = 0
                    self._entry_price = None
                    trades_to_remove.append(trade_id)
                    
                    remaining_order = tp_order_id if sl_filled else sl_order_id
                    if remaining_order and remaining_order in open_order_ids:
                        try:
                            cancel_order(token=self.token, account_id=self.account_id, 
                                        order_id=remaining_order)
                        except Exception as e:
                            logging.error(f"Failed to cancel order {remaining_order}: {e}")
                    
                    try:
                        online_learn_from_trade(trade)
                    except Exception as e:
                        logging.debug(f"Online learning failed: {e}")
            
            for trade_id in trades_to_remove:
                del self.open_trades[trade_id]
                
        except Exception as e:
            logging.error(f"Protective order check failed: {e}")
    
    def place_protective_orders(self, num_contracts, current_price, is_long, sl_mult, tp_mult, 
                              latest_features, order_id, linked_order_id):
        """Place protective orders with SMC-based logic."""
        try:
            if order_id in self.open_trades:
                existing = self.open_trades[order_id]
                if existing.get("sl_order_id") or existing.get("tp_order_id"):
                    return
            
            if is_long:
                stop_price = current_price * (1.0 - abs(sl_mult))
                limit_price = current_price * (1.0 + abs(tp_mult))
            else:
                stop_price = current_price * (1.0 + abs(sl_mult))
                limit_price = current_price * (1.0 - abs(tp_mult))
            
            stop_price = round(stop_price / TICK_SIZE) * TICK_SIZE
            limit_price = round(limit_price / TICK_SIZE) * TICK_SIZE
            
            exit_side = 1 if is_long else 0
            
            sl_order_id = None
            try:
                sl_order_id = place_order(
                    token=self.token,
                    account_id=self.account_id,
                    contract_id=self.contract_id,
                    order_type=4,  # Stop order
                    side=exit_side,
                    size=num_contracts,
                    stop_price=stop_price,
                    linked_order_id=linked_order_id
                )
            except Exception as e:
                logging.error(f"SL order placement failed: {e}")
            
            tp_order_id = None
            try:
                tp_order_id = place_order(
                    token=self.token,
                    account_id=self.account_id,
                    contract_id=self.contract_id,
                    order_type=1,  # Limit order
                    side=exit_side,
                    size=num_contracts,
                    limit_price=limit_price,
                    linked_order_id=linked_order_id
                )
            except Exception as e:
                logging.error(f"TP order placement failed: {e}")
            
            self.open_trades[linked_order_id] = {
                "entry_price": current_price,
                "sl_order_id": sl_order_id,
                "tp_order_id": tp_order_id,
                "is_long": bool(is_long),
                "size": num_contracts
            }
            
            logging.info(f"Protective orders placed: SL={stop_price}, TP={limit_price}")
            
        except Exception as e:
            logging.error(f"Protective order placement failed: {e}")

# --- MODE HANDLER ---
class ModeHandler:
    """Enhanced mode handler with unified data processing."""
    
    def __init__(self, mode='sim', account_id='', token=None):
        if mode not in ['sim', 'live']:
            raise ValueError("Mode must be either 'sim' or 'live'")
        
        self.mode = mode
        self.account_id = account_id
        self.token = token
        self.data_pipeline = DataPipeline(mode)
    
    def is_sim(self) -> bool:
        return self.mode == 'sim'
    
    def is_live(self) -> bool:
        return self.mode == 'live'
    
    def get_metrics_csv(self) -> str:
        return "sim_metrics.csv"
    
    def run_sim_mode(self, contract_id):
        """Run simulation mode with standardized data processing."""
        
        aggregator = BarAggregator()
        processor = Processor(self.account_id, contract_id, self.token, mode='sim')
        processor.aggregator = aggregator
        
        metrics = []
        processor.metrics_cb = metrics.append
        
        print("[SIM] Fetching historical data...")
        
        end = datetime.utcnow()
        start = end - timedelta(days=HIST_DAYS)
        
        bars = fetch_bars(
            token=self.token,
            contract_id=contract_id,
            start_time=start,
            end_time=end,
            unit=UNIT,
            unit_number=UNIT_NUMBER,
            limit=BAR_LIMIT,
            live=False
        )
        
        bars.reverse()  # Chronological order
        
        warmup_n = min(1000, max(120, len(bars) // 10))
        prebars = bars[:warmup_n]
        future_bars = bars[warmup_n:]
        
        def normalize_bars(bars_list):
            if not bars_list:
                return []
            df = pd.DataFrame(bars_list)
            df = df.rename(columns={c: c.lower() for c in df.columns})
            return df.to_dict('records')
        
        prebars = normalize_bars(prebars)
        future_bars = normalize_bars(future_bars)
        
        aggregator.replace_bars(prebars)
        
        print(f"[SIM] Processing {len(future_bars)} bars as ticks...")
        print(f"[SIM] Warmup: {len(prebars)} bars, Simulation: {len(future_bars)} bars")
        
        for i, bar in enumerate(future_bars):
            tick = {
                "timestamp": bar["timestamp"],
                "price": bar["close"],
                "volume": max(1, bar.get("volume", 1))
            }
            
            try:
                result = processor.process_tick(tick, aggregator)
                if i % 100 == 0:
                    print(f"[SIM] Processed {i+1}/{len(future_bars)} bars")
            except Exception as e:
                logging.error(f"Sim tick processing failed: {e}")
        
        try:
            if metrics:
                df_metrics = pd.DataFrame(metrics)
                if "timestamp" in df_metrics.columns:
                    df_metrics["timestamp"] = pd.to_datetime(df_metrics["timestamp"])
                    df_metrics = df_metrics.sort_values("timestamp")
                
                df_metrics.to_csv(self.get_metrics_csv(), index=False)
                print(f"[SIM] Saved {len(df_metrics)} metrics to {self.get_metrics_csv()}")
            else:
                print("[SIM] No metrics captured")
        except Exception as e:
            print(f"[SIM] Failed to save metrics: {e}")
        
        print(f"[SIM] Simulation complete - processed {len(future_bars)} total bars")
    
    def run_live_mode(self, contract_id=None):
        """Run live mode with enhanced data handling."""
        
        try:
            accounts = get_active_accounts(self.token)
            print(f"[LIVE] Found {len(accounts)} accounts")
        except Exception as e:
            print(f"[LIVE] Failed to fetch accounts: {e}")
            return
        
        lookup = str(self.account_id).strip().lower()
        matched_account = None
        
        for account in accounts:
            account_id = str(account.get('id', '')).lower()
            account_name = str(account.get('name', '')).lower()
            
            if lookup in [account_id, account_name]:
                matched_account = account
                break
        
        if not matched_account:
            print(f"[LIVE] No account found matching '{self.account_id}'")
            return
        
        resolved_account_id = matched_account.get('id')
        self.account_id = resolved_account_id
        print(f"[LIVE] Using account: {resolved_account_id}")
        
        if contract_id is None:
            contract_id = get_latest_contract_id(self.token, "ES")
        
        aggregator = BarAggregator()
        processor = Processor(self.account_id, contract_id, self.token, mode='live')
        processor.aggregator = aggregator
        
        print("[LIVE] Loading historical data...")
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)
        
        bars = fetch_bars(
            token=self.token,
            contract_id=contract_id,
            start_time=start_time,
            end_time=end_time,
            unit=UNIT,
            unit_number=UNIT_NUMBER,
            limit=300,
            live=False
        )
        
        bars_df = pd.DataFrame(bars[::-1])  # Chronological
        bars_df = bars_df.rename(columns={c: c.lower() for c in bars_df.columns})
        
        try:
            smc_bars = self._generate_smc_bars(bars_df)
            aggregator.replace_bars(smc_bars)
            print(f"[LIVE] Loaded {len(bars)} historical bars with SMC features")
        except Exception as e:
            aggregator.replace_bars(bars_df.to_dict('records'))
            print(f"[LIVE] Loaded {len(bars)} basic historical bars (SMC failed: {e})")
        
        if hasattr(self, 'client'):
            try:
                self.client.hub_connection.stop()
            except:
                pass
        
        self.client = SignalRClient(token=self.token, contract_id=contract_id, hub_type="market")
        
        def handle_tick(data):
            """Enhanced tick handler with validation."""
            try:
                if not isinstance(data, dict) or not data.get("price"):
                    return
                
                if data.get("type") == 6:
                    return
                
                trades = []
                if isinstance(data.get("arguments"), list) and len(data["arguments"]) > 1:
                    if isinstance(data["arguments"][1], list):
                        trades = data["arguments"][1]
                
                if trades:
                    for trade in trades:
                        if isinstance(trade, dict) and trade.get("price"):
                            processor.process_tick(trade, aggregator)
                else:
                    processor.process_tick(data, aggregator)
                    
            except Exception as e:
                logging.error(f"Live tick processing error: {e}")
        
        self.client.add_tick_listener(handle_tick)
        self.client.connect()
        
        print("[LIVE] Connected and awaiting market data...")
    
    def _generate_smc_bars(self, bars_df: pd.DataFrame) -> List[Dict]:
        """Generate SMC features for historical bars."""
        
        bars_df = bars_df.sort_values("timestamp").reset_index(drop=True)
        
        try:
            swings = smc.swing_highs_lows(bars_df)
            bos = smc.bos_choch(bars_df, swings)
            ob = smc.ob(bars_df, swings)
            fvg = smc.fvg(bars_df)
            liq = smc.liquidity(bars_df, swings)
            
            components = [bars_df, swings, bos, ob, fvg, liq]
            components = [df.loc[:, ~df.columns.duplicated()] for df in components]
            
            smc_df = pd.concat(components, axis=1)
            smc_df = smc_df.loc[:, ~smc_df.columns.duplicated()]
            
            latest_ts = bars_df["timestamp"].max()
            smc_df = smc_df[smc_df["timestamp"] <= latest_ts]
            
            return smc_df.to_dict('records')[::-1]  # Reverse for chronological order
            
        except Exception as e:
            logging.error(f"SMC generation failed: {e}")
            return bars_df.to_dict('records')[::-1]
    
    def kill_live(self):
        """Stop live mode."""
        if hasattr(self, 'client'):
            try:
                self.client.hub_connection.stop()
                print("[LIVE] Connection stopped")
            except Exception as e:
                print(f"[LIVE] Error stopping connection: {e}")

__all__ = ["ModeHandler", "BarAggregator", "Processor", "DataPipeline", "FeatureValidator"]

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Credentials
    username = "not1me"
    api_key = "DaIgCHxvhzqAI+vNEJk6COxqjGwq6dcrMo/jOMIt/AY="
    token = login_and_get_token(username, api_key)
    contract_id = get_latest_contract_id(token, "ES")
    
    # Mode selection
    mode_type = os.getenv("TRADING_MODE", "sim").lower()
    account_id = os.getenv("ACCOUNT_ID", "PRAC-V2-229988-46642281")
    
    if mode_type == "sim":
        handler = ModeHandler(mode="sim", token=token)
        handler.run_sim_mode(contract_id)
    elif mode_type == "live":
        live_handler = ModeHandler(mode="live", account_id=account_id, token=token)
        live_handler.run_live_mode(contract_id)
