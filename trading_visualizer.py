# trading_visualizer.py - Real-time visualization of your trading bot
"""
Trading Visualizer expects:
- prediction dict: {
    'p_take': float in [0,1],
    'p_conf': float in [0,1],
    'direction': 'LONG'|'SHORT'|'NONE',
    'tp_offset': float (fractional target, e.g., 0.02 = +2%),
    'sl_offset': float (fractional stop, e.g., 0.01 = 1%)
}
- feature vector: list[float] whose first 19 positions map to names in FEATURE_NAMES below.
Use nn_adapter(raw) to convert raw NN tensors to this dict.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')
import logging, os
import inspect, importlib, sys, json, textwrap
FEATURE_NAMES = [
    "open","high","low","close","volume",
    "HighLow","Level","BOS","CHOCH","OB","FVG","Liquidity",
    "HighLow_pos","HighLow_neg","TimeSinceBOS","TimeSinceOB",
    "TimeSinceFVG","TimeSinceLiquidity","BarIndex"
]

def nn_adapter(raw):
    """Convert model raw outputs to visualizer dict.
    raw can be a tuple/list/np array/torch tensor with 6 outputs in order:
    [p_take, p_conf, tp_offset, sl_offset, tp_quality, sl_quality].
    Returns dict for add_data_point().
    """
    try:
        # Lazy import torch to support numpy-only usage
        try:
            import torch
            to_float = lambda x: float(x.detach().cpu().item()) if hasattr(x, 'detach') else float(x)
        except Exception:
            to_float = float
        p_take, p_conf, tp_off, sl_off, *_ = [to_float(v) for v in list(raw)]
        direction = 'LONG' if tp_off >= 0 else 'SHORT'
        return {
            'p_take': max(0.0, min(1.0, p_take)),
            'p_conf': max(0.0, min(1.0, p_conf)),
            'direction': direction,
            'tp_offset': abs(tp_off),
            'sl_offset': abs(sl_off),
        }
    except Exception as e:
        logging.debug(f"nn_adapter failed: {e}")
        return {'p_take':0.0,'p_conf':0.0,'direction':'NONE','tp_offset':0.0,'sl_offset':0.0}

class TradingVisualizer:
    def __init__(self, max_bars=500):
        self.max_bars = max_bars
        self.data_buffer = deque(maxlen=max_bars)
        self.prediction_buffer = deque(maxlen=max_bars)
        self.last_features = []
        self.feature_changes = {}
        # Set up the plot
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(4, 1, figsize=(15, 12))
        self.fig.suptitle('Trading Bot Real-Time Analysis', fontsize=16, color='white')
        # Initialize plots
        self._setup_plots()
        
    def _setup_plots(self):
        # Plot 1: Price + SMC Indicators
        self.ax_price = self.axes[0]
        self.ax_price.set_title('ES Price + SMC Signals', color='white')
        self.ax_price.grid(True, alpha=0.3)
        # Plot 2: Model Predictions
        self.ax_pred = self.axes[1] 
        self.ax_pred.set_title('Model Predictions', color='white')
        self.ax_pred.grid(True, alpha=0.3)
        self.ax_pred.set_ylim(0, 1)
        # Plot 3: Feature Values
        self.ax_features = self.axes[2]
        self.ax_features.set_title('Key Feature Values', color='white')
        self.ax_features.grid(True, alpha=0.3)
        # Plot 4: Trade Signals
        self.ax_signals = self.axes[3]
        self.ax_signals.set_title('Trade Signals & P&L', color='white')
        self.ax_signals.grid(True, alpha=0.3)
        
    def add_data_point(self, price_data, smc_data, prediction, features):
        """Add new data point to visualization"""
        timestamp = price_data.get('timestamp', datetime.now())

        # Normalize timestamp
        try:
            if not isinstance(timestamp, datetime):
                timestamp = pd.to_datetime(timestamp)
        except Exception:
            timestamp = datetime.now()

        # Normalize SMC dict with defaults
        smc_defaults = {'BOS':0.0,'CHOCH':0.0,'OB':0.0,'FVG':0.0,'Liquidity':0.0,'HighLow':0.0}
        smc_data = {**smc_defaults, **(smc_data or {})}

        # Store data
        data_point = {
            'timestamp': timestamp,
            'open': price_data.get('open'),
            'high': price_data.get('high'), 
            'low': price_data.get('low'),
            'close': price_data.get('close'),
            'volume': price_data.get('volume', 0),
            **smc_data  # BOS, CHOCH, OB, FVG, Liquidity, etc.
        }

        pred_point = {
            'timestamp': timestamp,
            'p_take': prediction.get('p_take', 0),
            'p_conf': prediction.get('p_conf', 0),
            'direction': prediction.get('direction', 'NONE'),
            'tp_offset': prediction.get('tp_offset', 0),
            'sl_offset': prediction.get('sl_offset', 0)
        }

        self.data_buffer.append(data_point)
        self.prediction_buffer.append(pred_point)

        # Store key features for debugging
        if hasattr(self, 'last_features'):
            self.feature_changes = self._compare_features(self.last_features, features)
        self.last_features = list(features) if isinstance(features, (list, tuple, np.ndarray)) else []
        
    def _compare_features(self, old_features, new_features):
        """Compare feature vectors to detect changes"""
        if not old_features or not new_features or len(old_features) != len(new_features):
            return {}
        
        changes = {}
        feature_names = [
            "open", "high", "low", "close", "volume",
            "HighLow", "Level", "BOS", "CHOCH", "OB", "FVG", "Liquidity", 
            "HighLow_pos", "HighLow_neg",
            "TimeSinceBOS", "TimeSinceOB", "TimeSinceFVG", "TimeSinceLiquidity",
            "BarIndex"
        ]
        
        for i, (old, new) in enumerate(zip(old_features, new_features)):
            if abs(old - new) > 1e-6:  # Detect meaningful changes
                name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                changes[name] = {'old': old, 'new': new, 'change': new - old}
                
        return changes
        
    def update_plots(self):
        """Update all visualization plots"""
        try:
            if len(self.data_buffer) < 2:
                return
            for ax in self.axes:
                ax.clear()
            df_data = pd.DataFrame(list(self.data_buffer))
            df_pred = pd.DataFrame(list(self.prediction_buffer))
            # Forward fill minimal missing values for plotting
            df_data = df_data.ffill().bfill()
            df_pred = df_pred.ffill().bfill()
            timestamps = df_data['timestamp']
            self._plot_price_and_smc(df_data, timestamps)
            self._plot_predictions(df_pred, timestamps)
            self._plot_features(df_data)
            self._plot_trade_signals(df_data, df_pred, timestamps)
            plt.tight_layout()
            self.fig.canvas.draw_idle()
        except Exception as e:
            logging.debug(f"update_plots failed: {e}")
        
    def _plot_price_and_smc(self, df_data, timestamps):
        ax = self.axes[0]
        ax.set_title('ES Price + SMC Signals', color='white')
        # Candlestick-like price plot
        closes = df_data['close'].values
        highs = df_data['high'].values
        lows = df_data['low'].values
        ax.plot(timestamps, closes, 'white', linewidth=1, label='Close')
        ax.fill_between(timestamps, lows, highs, alpha=0.2, color='gray')
        # SMC Signals as colored dots
        for i, (ts, row) in enumerate(zip(timestamps, df_data.itertuples())):
            y_pos = row.close
            if hasattr(row, 'BOS') and abs(row.BOS) > 0.1:
                color = 'lime' if row.BOS > 0 else 'red'
                ax.scatter(ts, y_pos, c=color, s=100, marker='^', label='BOS' if i == 0 else "")
            if hasattr(row, 'OB') and abs(row.OB) > 0.1:
                color = 'cyan' if row.OB > 0 else 'orange' 
                ax.scatter(ts, y_pos, c=color, s=80, marker='s', alpha=0.7)
            if hasattr(row, 'Liquidity') and abs(row.Liquidity) > 0.1:
                ax.scatter(ts, y_pos, c='yellow', s=60, marker='*', alpha=0.8)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc='upper left')
        
    def _plot_predictions(self, df_pred, timestamps):
        ax = self.axes[1]
        ax.set_title('Model Predictions Over Time', color='white')

        if len(df_pred) > 0:
            # Plot series
            p_take_values = df_pred['p_take'].values
            p_conf_values = df_pred['p_conf'].values
            ax.plot(timestamps, p_take_values, 'lime', linewidth=2)
            ax.plot(timestamps, p_conf_values, 'cyan', linewidth=2)

            # Rolling flatness detector (windowed uniqueness)
            win = min(50, len(p_take_values))
            if win >= 5:
                rtake = pd.Series(np.round(p_take_values, 4)).rolling(win).apply(lambda x: len(np.unique(x)), raw=True)
                rconf = pd.Series(np.round(p_conf_values, 4)).rolling(win).apply(lambda x: len(np.unique(x)), raw=True)
                last_utake = int(rtake.iloc[-1]) if not np.isnan(rtake.iloc[-1]) else 0
                last_uconf = int(rconf.iloc[-1]) if not np.isnan(rconf.iloc[-1]) else 0
                # Legends show last-window uniqueness
                ax.legend([f'P(Take) – {last_utake} uniq @{win}', f'P(Conf) – {last_uconf} uniq @{win}'])
                # Highlight if current window is flat
                if last_utake <= 2:
                    ax.text(0.5, 0.8, 'FLAT P(TAKE) IN LAST WINDOW',
                            transform=ax.transAxes, color='red', fontsize=12, weight='bold',
                            ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.25))
                if last_uconf <= 2:
                    ax.text(0.5, 0.7, 'FLAT P(CONF) IN LAST WINDOW',
                            transform=ax.transAxes, color='orange', fontsize=11, weight='bold',
                            ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.25))
            else:
                ax.legend(['P(Take)', 'P(Conf)'])

        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
    def _plot_features(self, df_data):
        ax = self.axes[2]
        ax.set_title('Key SMC Features', color='white')
        feature_cols = ['BOS', 'CHOCH', 'OB', 'FVG', 'Liquidity', 'HighLow']
        colors = ['red', 'orange', 'yellow', 'lime', 'cyan', 'magenta']
        for i, col in enumerate(feature_cols):
            if col in df_data.columns:
                values = df_data[col].values
                non_zero_mask = np.abs(values) > 0.01
                if np.any(non_zero_mask):
                    ax.scatter(df_data['timestamp'][non_zero_mask], values[non_zero_mask], 
                             c=colors[i], label=f'{col} ({np.sum(non_zero_mask)} signals)', s=50, alpha=0.8)
                else:
                    ax.text(0.02, 0.95 - i*0.1, f'{col}: NO SIGNALS', 
                           transform=ax.transAxes, color=colors[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc='upper right')
        
    def _plot_trade_signals(self, df_data, df_pred, timestamps):
        ax = self.axes[3]
        ax.set_title('Trade Signals & Entry Points', color='white')
        if len(df_pred) > 0:
            # Simulate trade signals based on thresholds
            take_threshold = 0.7
            conf_threshold = 0.7
            # Show thresholds as label for clarity
            ax.text(0.02, 0.75, f'Thresh: take {take_threshold:.2f}, conf {conf_threshold:.2f}',
                    transform=ax.transAxes, color='white', fontsize=9, alpha=0.7)
            trade_signals = (df_pred['p_take'] >= take_threshold) & (df_pred['p_conf'] >= conf_threshold)
            # Plot price
            ax.plot(timestamps, df_data['close'], 'white', alpha=0.7, linewidth=1)
            # Mark potential trade entries
            for i, (should_trade, ts, pred_row, price_row) in enumerate(zip(
                trade_signals, timestamps, df_pred.itertuples(), df_data.itertuples())):
                if should_trade:
                    color = 'lime' if hasattr(pred_row, 'direction') and 'LONG' in str(pred_row.direction) else 'red'
                    ax.scatter(ts, price_row.close, c=color, s=150, marker='^',
                               edgecolors='white', linewidth=2, alpha=0.9, facecolors=color)
                    # Show TP/SL levels with correct directional math
                    if hasattr(pred_row, 'tp_offset') and hasattr(pred_row, 'sl_offset'):
                        dir_str = str(getattr(pred_row, 'direction', 'NONE')).upper()
                        tp_mag = abs(float(pred_row.tp_offset))
                        sl_mag = abs(float(pred_row.sl_offset))
                        if 'LONG' in dir_str:
                            tp_price = price_row.close * (1 + tp_mag)
                            sl_price = price_row.close * (1 - sl_mag)
                        elif 'SHORT' in dir_str:
                            tp_price = price_row.close * (1 - tp_mag)
                            sl_price = price_row.close * (1 + sl_mag)
                        else:
                            # Fallback: assume magnitudes with SL below TP around current price
                            tp_price = price_row.close * (1 + tp_mag)
                            sl_price = price_row.close * (1 - sl_mag)
                        low = min(sl_price, tp_price)
                        high = max(sl_price, tp_price)
                        ax.plot([ts, ts], [low, high], color=color, alpha=0.5, linewidth=3)
            # Show statistics
            total_signals = np.sum(trade_signals)
            ax.text(0.02, 0.95, f'Trade Signals: {total_signals}',
                   transform=ax.transAxes, color='yellow', fontsize=12, weight='bold')
            if hasattr(self, 'feature_changes') and self.feature_changes:
                changes_text = f"Features Changed: {len(self.feature_changes)}"
                ax.text(0.02, 0.85, changes_text,
                       transform=ax.transAxes, color='cyan', fontsize=10)
        ax.grid(True, alpha=0.3)
        
    def show_feature_debug(self):
        """Show detailed feature debugging info"""
        if hasattr(self, 'last_features') and self.last_features:
            feature_names = FEATURE_NAMES
            print("\n=== CURRENT FEATURE VALUES ===")
            for i, name in enumerate(feature_names[:len(self.last_features)]):
                value = self.last_features[i]
                status = "📈" if abs(value) > 0.01 else "💤"
                print(f"{status} {name:15s}: {value:10.6f}")
        if hasattr(self, 'feature_changes') and self.feature_changes:
            print("\n=== RECENT FEATURE CHANGES ===")
            for name, change_info in self.feature_changes.items():
                print(f"🔄 {name}: {change_info['old']:.6f} → {change_info['new']:.6f} (Δ{change_info['change']:+.6f})")
        else:
            print("\n⚠️  NO FEATURE CHANGES DETECTED - This might explain flat predictions!")

 # === Explanatory report generator ===

def explain_algo(output_format="markdown", include_code=True, forward_probe=True):
    """Generate a detailed, end-to-end explanation of the algo stack.

    output_format: "markdown" | "json"
    include_code: include function sources where available
    forward_probe: run a synthetic forward to validate I/O contract if torch is present
    """
    report = {
        "overview": {
            "purpose": "Sequence model for trade decisioning with SMC-driven features and multitask outputs",
            "modules": ["nn_model", "trading_visualizer"],
            "io_contract": {
                "sequence_len": 120,
                "feature_count": len(FEATURE_NAMES)  # base visible features; model may expect more
            },
            "outputs_order": ["p_take","p_conf","tp_offset","sl_offset","tp_quality","sl_quality"],
            "decision_policy": {
                "take_threshold": 0.7,
                "conf_threshold": 0.7,
                "direction_logic": "tp_offset sign (>=0 LONG, <0 SHORT)",
                "offsets": "fractional magnitudes applied around current price"
            }
        },
        "features": {
            "visible_names": FEATURE_NAMES
        },
        "model": {},
        "data_pipeline": {},
        "training": {},
        "visualization": {
            "adapter": "nn_adapter(raw) coerces tensors/arrays → dict with bounded probs and positive offsets",
            "plots": [
                "Price + SMC markers (BOS/OB/Liquidity)",
                "Predictions with flatness detector",
                "Key SMC features scatter",
                "Trade entries with TP/SL brackets"
            ]
        },
        "failure_modes": [
            "Flat probabilities due to class imbalance or scaler misfit",
            "Schema drift: missing SMC columns or wrong order",
            "NaNs/Infs in features causing silent masking",
            "Mismatched output ordering vs adapter",
            "Visualization thresholds diverge from live execution thresholds"
        ]
    }

    # Try to import nn_model
    m = None
    try:
        m = importlib.import_module("nn_model")
    except Exception as e:
        report["model"]["import_error"] = str(e)

    # Introspect model meta
    if m:
        # Feature columns vs FEATURE_NAMES
        feat_cols = []
        if hasattr(m, "feature_columns"):
            try:
                feat_cols = list(m.feature_columns())
            except Exception as e:
                report["features"]["feature_columns_error"] = str(e)
        report["features"]["model_feature_columns_count"] = len(feat_cols) if feat_cols else None
        if feat_cols:
            report["features"]["model_feature_columns_preview"] = feat_cols[:50]

        # Hyperparams
        for k in ["DEFAULT_DROPOUT","DEFAULT_LR","TP_SCALE","SL_SCALE"]:
            if hasattr(m, k):
                report.setdefault("model",{})[k.lower()] = getattr(m, k)

        # Architecture
        ModelCls = getattr(m, "Model", None)
        if ModelCls:
            try:
                model = ModelCls(input_size=len(feat_cols) if feat_cols else len(FEATURE_NAMES))
                # Param counts
                try:
                    import torch
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                except Exception:
                    total_params = trainable_params = None
                report["model"]["architecture"] = {
                    "class": ModelCls.__name__,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "str": str(model)
                }
                # Forward probe
                if forward_probe:
                    try:
                        import torch
                        x = torch.randn(2, 120, (len(feat_cols) if feat_cols else len(FEATURE_NAMES)))
                        with torch.no_grad():
                            out = model(x)
                        if isinstance(out, (list, tuple)):
                            shapes = [tuple(getattr(t, "shape", ())) for t in out]
                        else:
                            shapes = [tuple(getattr(out, "shape", ()))]
                        report["model"]["forward_probe_shapes"] = shapes
                    except Exception as e:
                        report["model"]["forward_probe_error"] = str(e)
            except Exception as e:
                report["model"]["init_error"] = str(e)

        # Data pipeline: _ensure_smc_context and helpers
        for fname in ["_ensure_smc_context", "prepare_data", "train_model"]:
            if hasattr(m, fname):
                fn = getattr(m, fname)
                entry = {"signature": str(inspect.signature(fn))}
                try:
                    entry["doc"] = inspect.getdoc(fn)
                except Exception:
                    pass
                if include_code:
                    try:
                        entry["source"] = textwrap.dedent(inspect.getsource(fn))
                    except Exception:
                        entry["source"] = None
                report["data_pipeline" if fname!="train_model" else "training"][fname] = entry

        # Loss weights and calibration hints
        loss_keys = ["alpha_take","alpha_conf","alpha_tp","alpha_sl","alpha_tpq","alpha_slq","pos_weight"]
        loss_block = {}
        for k in loss_keys:
            for key in (k, k.upper()):
                if hasattr(m, key):
                    loss_block[k] = getattr(m, key)
                    break
        if loss_block:
            report["training"]["loss_weights"] = loss_block

    # Render
    if output_format == "json":
        return json.dumps(report, indent=2, default=str)

    # Markdown renderer
    md = []
    o = report["overview"]
    md.append("# Algo System Overview")
    md.append(f"Sequence length: **{o['io_contract']['sequence_len']}** | Feature count (visible): **{o['io_contract']['feature_count']}**")
    md.append(f"Outputs order: `{' ,'.join(o['outputs_order'])}`")
    md.append("\n## Decision Policy")
    md.append(f"- Take if `p_take ≥ {o['decision_policy']['take_threshold']}` and `p_conf ≥ {o['decision_policy']['conf_threshold']}`")
    md.append(f"- Direction: {o['decision_policy']['direction_logic']}")
    md.append(f"- Offsets: {o['decision_policy']['offsets']}")

    md.append("\n## Features")
    md.append("Visible feature names:")
    md.append("```")
    md.append("\n".join(FEATURE_NAMES))
    md.append("```")
    if report["features"].get("model_feature_columns_preview"):
        md.append("Model-declared features (preview):")
        md.append("```")
        md.append("\n".join(report["features"]["model_feature_columns_preview"]))
        md.append("```")

    md.append("\n## Model")
    if report.get("model",{}).get("architecture"):
        arch = report["model"]["architecture"]
        md.append(f"Class: **{arch['class']}** | Params: **{arch['trainable_params']}** / {arch['total_params']}")
        md.append("\n### Torch Module String\n")
        md.append("```")
        md.append(arch["str"])
        md.append("```")
    else:
        md.append("Model architecture not available.")

    md.append("\n## Data Pipeline")
    for k, v in report.get("data_pipeline",{}).items():
        md.append(f"### {k}{v.get('signature','')}")
        if v.get("doc"):
            md.append(v["doc"]) 
        if include_code and v.get("source"):
            md.append("```")
            md.append(v["source"])
            md.append("```")

    md.append("\n## Training")
    if report.get("training",{}).get("loss_weights"):
        md.append("Loss weights:")
        md.append("```")
        md.append(json.dumps(report["training"]["loss_weights"], indent=2))
        md.append("```")
    if report.get("training",{}).get("train_model"):
        v = report["training"]["train_model"]
        md.append(f"### train_model{v.get('signature','')}")
        if v.get("doc"): md.append(v["doc"])
        if include_code and v.get("source"):
            md.append("```")
            md.append(v["source"])
            md.append("```")

    md.append("\n## Visualization")
    md.append("Adapter contract and plots described above. Thresholds are in `_plot_trade_signals`.\n")

    md.append("\n## Known Failure Modes")
    for fm in report["failure_modes"]:
        md.append(f"- {fm}")

    return "\n".join(md)

# Integration with your existing code
def integrate_visualizer_with_processor():
    """
    Add this to your Processor class in mode_handler.py
    """
    integration_code = '''
# Add near top-level of your module:
from trading_visualizer import TradingVisualizer, nn_adapter
import os, logging

# Add to Processor.__init__():
self.visualizer = TradingVisualizer() if os.getenv("ENABLE_VIZ", "false").lower() == "true" else None

# Modify process_tick method around your prediction site:
# ... after you compute `sequence`, `new_bar`, `smc_df`, and raw model outputs `raw_pred` ...
if self.visualizer and new_bar:
    try:
        latest_bar = aggregator.latest_bar()  # dict with open/high/low/close/volume/timestamp
        smc_latest = smc_df.iloc[-1].to_dict()
        pred_dict  = nn_adapter(raw_pred)
        self.visualizer.add_data_point(latest_bar, smc_latest, pred_dict, sequence[-1])
        if getattr(self, "_processed_tick_count", 0) % 10 == 0:
            self.visualizer.update_plots()
            plt.pause(0.01)
    except Exception as e:
        logging.debug(f"Visualization failed: {e}")
'''
    return integration_code

# Standalone testing function  
def test_visualizer_with_sample_data():
    """Test the visualizer with sample data"""
    np.random.seed(42)
    viz = TradingVisualizer()
    # Generate sample data to test visualization
    base_price = 4500.0
    timestamps = pd.date_range(start='2024-01-01 09:30:00', periods=100, freq='1min')
    for i, ts in enumerate(timestamps):
        # Sample price data
        price_change = np.random.normal(0, 0.5)
        base_price += price_change
        price_data = {
            'timestamp': ts,
            'open': base_price - 0.25,
            'high': base_price + np.random.uniform(0, 1),
            'low': base_price - np.random.uniform(0, 1), 
            'close': base_price,
            'volume': np.random.randint(50, 200)
        }
        # Sample SMC data
        smc_data = {
            'BOS': np.random.choice([0, 1, -1], p=[0.9, 0.05, 0.05]),
            'CHOCH': np.random.choice([0, 1, -1], p=[0.95, 0.025, 0.025]),
            'OB': np.random.choice([0, 1, -1], p=[0.85, 0.075, 0.075]),
            'FVG': np.random.choice([0, 1, -1], p=[0.9, 0.05, 0.05]),
            'Liquidity': np.random.choice([0, 1], p=[0.95, 0.05]),
            'HighLow': np.random.choice([0, 1, -1], p=[0.9, 0.05, 0.05])
        }
        # Sample predictions - mix flat and varied to test detection
        if i < 50:  # First half flat (simulating your bug)
            prediction = {
                'p_take': 0.746,
                'p_conf': 0.557,
                'direction': 'LONG',
                'tp_offset': 0.04825,
                'sl_offset': 0.02779
            }
        else:  # Second half varied
            prediction = {
                'p_take': np.random.uniform(0.3, 0.9),
                'p_conf': np.random.uniform(0.2, 0.8),
                'direction': np.random.choice(['LONG', 'SHORT']),
                'tp_offset': np.random.uniform(0.01, 0.05),
                'sl_offset': np.random.uniform(0.01, 0.03)
            }
        # Sample features
        features = np.random.normal(0, 1, 19).tolist()
        viz.add_data_point(price_data, smc_data, prediction, features)
        if i % 20 == 0:  # Update every 20 points
            viz.update_plots()
            plt.pause(0.1)
    viz.show_feature_debug()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trading Visualizer and Algo Explainer")
    parser.add_argument("--explain", action="store_true", help="Print a detailed explanation of the algo")
    parser.add_argument("--json", action="store_true", help="Output explanation in JSON format")
    args = parser.parse_args()

    if args.explain:
        fmt = "json" if args.json else "markdown"
        print(explain_algo(output_format=fmt, include_code=True, forward_probe=True))
    else:
        os.environ.setdefault("ENABLE_VIZ", "true")
        print("Testing Trading Visualizer...")
        test_visualizer_with_sample_data()