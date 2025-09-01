# streamlit_app.py
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

# ---- Safe utility fallbacks (defined only if missing) ----
if 'build_activation_atlas' not in globals():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    def build_activation_atlas(model, sample_inputs, layer_name='fc1', n_clusters=8):
        import pandas as pd
        import numpy as np
        import torch
        # ensure 2D input
        X = np.asarray(sample_inputs, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        # capture activations via hook
        acts = {}
        layer = getattr(model, layer_name, None)
        if layer is None:
            return pd.DataFrame({'x': [], 'y': [], 'cluster': []})
        def _hook(m,i,o):
            acts['A'] = o.detach().cpu().numpy()
        h = layer.register_forward_hook(_hook)
        with torch.no_grad():
            _ = model(torch.tensor(X, dtype=torch.float32))
        h.remove()
        A = acts.get('A', np.zeros((len(X), 1)))
        # reduce and cluster
        if A.shape[1] > 2:
            A2 = PCA(n_components=2).fit_transform(A)
        else:
            # pad to 2 dims if needed
            pad = np.zeros((A.shape[0], 2 - A.shape[1])) if A.shape[1] < 2 else np.zeros((A.shape[0], 0))
            A2 = np.hstack([A, pad])
        k = min(max(2, n_clusters), max(2, len(A2)))
        labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(A2) if len(A2) >= k else np.zeros(len(A2), dtype=int)
        return pd.DataFrame({'x': A2[:,0], 'y': A2[:,1], 'cluster': labels})

if 'activation_timeline' not in globals():
    import matplotlib.pyplot as plt
    def activation_timeline(model, sample_inputs, layer_name='fc1'):
        import numpy as np, torch
        X = np.asarray(sample_inputs, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        acts = {}
        layer = getattr(model, layer_name, None)
        if layer is None:
            fig, ax = plt.subplots(); ax.text(0.5,0.5,'Layer not found', ha='center'); return fig
        def _hook(m,i,o):
            acts['A'] = o.detach().cpu().numpy()
        h = layer.register_forward_hook(_hook)
        with torch.no_grad():
            _ = model(torch.tensor(X, dtype=torch.float32))
        h.remove()
        A = acts.get('A')
        fig, ax = plt.subplots()
        if A is None:
            ax.text(0.5,0.5,'No activations', ha='center')
        else:
            ax.imshow(A.T, aspect='auto')
            ax.set_xlabel('Sample'); ax.set_ylabel('Unit index')
        return fig

if 'feature_maximization' not in globals():
    def feature_maximization(model, layer_name, neuron_idx, input_dim, steps=80, lr=0.05):
        import torch, numpy as np
        x = torch.zeros((1, input_dim), dtype=torch.float32, requires_grad=True)
        layer = getattr(model, layer_name, None)
        if layer is None:
            return np.zeros((input_dim,), dtype=float)
        acts = {}
        def _hook(m,i,o):
            acts['A'] = o
        h = layer.register_forward_hook(_hook)
        opt = torch.optim.SGD([x], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            y = model(x)
            A = acts.get('A')
            if A is None:
                break
            loss = -A[0, neuron_idx]
            loss.backward()
            opt.step()
        h.remove()
        return x.detach().cpu().numpy().flatten()

# --- Brain scan functions ---
from datetime import datetime, timedelta

from mode_handler import ModeHandler, UNIT, UNIT_NUMBER, BAR_LIMIT
from signalr import login_and_get_token, search_contracts, fetch_bars, get_active_accounts
from nn_model import load_model_and_predict, AdvancedSignalPredictor

# ── Persistent Session State Initialization ──────────────────────────────
if "live_mode" not in st.session_state:
    st.session_state["live_mode"] = False
if "live_running" not in st.session_state:
    st.session_state["live_running"] = False

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("Configuration")

username    = st.sidebar.text_input("Username", value="not1me", key="username")
api_key     = st.sidebar.text_input("API Key", value="ae+1hspMIKKqsYH++C9RwTebW374ubYc+OxS5uLDbfQ=", type="password", key="api_key")
search_text = st.sidebar.text_input("Contract search", value="ES")
mode        = st.sidebar.radio("Mode", ("sim", "live"))
account_name = st.sidebar.text_input("Account Name", value="PRAC-V2-229988-46642281", key="account_name")

lookback    = st.sidebar.number_input("Lookback days", min_value=1, max_value=365, value=30)

# --- Cost model & instrument (for Sim Metrics tab EV computation) ---
with st.sidebar.expander("Cost model & instrument", expanded=False):
    es_tick_size = st.number_input("Tick size", min_value=0.01, max_value=1.0, value=0.25, step=0.01, help="ES tick size")
    spread_ticks = st.number_input("Avg spread (ticks)", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
    slippage_ticks = st.number_input("Avg slippage (ticks)", min_value=0.0, max_value=5.0, value=0.5, step=0.25)
    fees_per_round_ticks = st.number_input("Fees per round (ticks)", min_value=0.0, max_value=5.0, value=0.1, step=0.05)
    total_cost_ticks = spread_ticks + slippage_ticks + fees_per_round_ticks

# ── Helpers ────────────────────────────────────────────────────────────────
@st.cache_data()
def get_token(u, k):
    return login_and_get_token(u, k)

@st.cache_data()
def find_contract(tkn, text, live=False):
    ctrs = search_contracts(tkn, text, live=live)
    print(f"[DEBUG] Searching for: '{text}' | Live: {live}")
    print(f"[DEBUG] Contracts found: {json.dumps(ctrs, indent=2)}")

    # Fallback to non-live search if live returned nothing
    if live and not ctrs:
        print("[DEBUG] Live search returned nothing, retrying with live=False")
        ctrs = search_contracts(tkn, text, live=False)
        print(f"[DEBUG] Fallback Contracts: {json.dumps(ctrs, indent=2)}")

    return ctrs[0]["id"] if ctrs else None
    
def load_trade_log():
    if os.path.exists("trade_performance.json"):
        try:
            with open("trade_performance.json") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except json.JSONDecodeError:
            print("Warning: trade_performance.json is empty or malformed.")
            return pd.DataFrame()
    return pd.DataFrame()

def load_training_history():
    if os.path.exists("training_history.json"):
        try:
            with open("training_history.json") as f:
                hist = json.load(f)
        except json.JSONDecodeError:
            print("Warning: training_history.json is empty or malformed.")
            return pd.DataFrame()

        # Require 'features' and 'labels' keys
        if not isinstance(hist, dict) or "features" not in hist or "labels" not in hist:
            print("Error: training_history.json missing 'features' or 'labels' keys.")
            return pd.DataFrame()

        df = pd.DataFrame(hist["features"])
        label_series = pd.Series(hist["labels"])
        if not df.empty and len(label_series) == len(df):
            df["label"] = label_series.apply(lambda x: x[0] if isinstance(x, (list, tuple)) and x else None)
            return df
        else:
            print("Warning: feature-label mismatch or empty DataFrame.")
            return pd.DataFrame()
    return pd.DataFrame()

# Stub for loading training metrics
def load_training_history_metrics():
    if os.path.exists("training_metrics.json"):
        with open("training_metrics.json") as f:
            metrics = json.load(f)
        df = pd.DataFrame(metrics)
        if 'epoch' not in df.columns:
            df = df.reset_index().rename(columns={df.columns[0]: 'epoch'})
        return df
    return pd.DataFrame()

# --- Helper to load simulation metrics (NN output probabilities, etc) ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
metrics_path = os.path.join(BASE_DIR, "sim_metrics.csv")

@st.cache_data()
def load_sim_metrics(path: str = metrics_path) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Normalize column names we expect from ModeHandler metrics callback
        cols = {c.lower(): c for c in df.columns}
        # Ensure a proper timestamp as datetime
        ts_col = None
        for candidate in ["timestamp", "time", "ts"]:
            if candidate in cols:
                ts_col = cols[candidate]
                break
        if ts_col is None:
            return pd.DataFrame()
        df[ts_col] = pd.to_datetime(df[ts_col])
        # Standardize names for downstream plotting
        rename_map = {}
        if "p_take" in df.columns:
            rename_map["p_take"] = "p_take"
        elif "take" in df.columns:
            rename_map["take"] = "p_take"
        if "p_conf" in df.columns:
            rename_map["p_conf"] = "p_conf"
        elif "confidence" in df.columns:
            rename_map["confidence"] = "p_conf"
        if "dir" in df.columns:
            rename_map["dir"] = "direction"
        elif "direction" in df.columns:
            rename_map["direction"] = "direction"
        if "price" in df.columns:
            rename_map["price"] = "price"
        df = df.rename(columns=rename_map)
        # Sort by time
        df = df.sort_values(ts_col).reset_index(drop=True)
        df = df.rename(columns={ts_col: "timestamp"})
        return df
    except Exception:
        return pd.DataFrame()

# ── Main ──────────────────────────────────────────────────────────────────
st.title("📊 SMC Trading Bot Dashboard")

if not api_key:
    st.warning("Enter your API key in the sidebar to begin.")
    st.stop()

token = get_token(username, api_key)
if token is None:
    st.error("Login failed. Please check your API key and username.")
    st.stop()

st.markdown(f"**Token:** `{token[:8]}...`")

# Resolve the numeric account ID dynamically using the valid token
accounts = get_active_accounts(token)
# Display debug output for account keys
# st.write(f"Debug accounts: {accounts}")
# Match by name (case-insensitive)
matching_accounts = [acct for acct in accounts if account_name.lower() in acct.get("name", acct.get("accountName", "")).lower()]
if not matching_accounts:
    st.error(f"No account found matching '{account_name}'.")
    st.stop()
first_acct = matching_accounts[0]
# Support different key names for the account ID
account_id = first_acct.get("accountId") or first_acct.get("id") or first_acct.get("AccountId") or first_acct.get("accountid")
if account_id is None:
    st.error(f"Account record missing an ID field: keys = {list(first_acct.keys())}")
    st.stop()
# Support different key names for the account display name
acct_name = first_acct.get("name") or first_acct.get("accountName") or str(account_id)
st.markdown(f"**Account:** `{acct_name}` (ID: {account_id})")

contract_id = find_contract(token, search_text, live=(mode=="live"))
if not contract_id:
    st.error(f"No contract matches '{search_text}'.")
    st.stop()

st.markdown(f"**Contract:** `{contract_id}`")

from mode_handler import ModeHandler

# Always refresh the ModeHandler to ensure the latest methods (like train_model) exist
st.session_state["handler"] = ModeHandler(
    mode=mode,
    account_id=account_id,
    token=token
)
handler = st.session_state["handler"]

# ── Manual retrain button ──────────────────────────────────────────────────
if st.button("⚙️ Train Model"):
    if not api_key:
        st.warning("Enter your API key in the sidebar to retrain.")
    else:
        from nn_model import train_model, convert_to_trades, prepare_data
        with st.spinner("Training model…"):
            try:
                # Fetch historical bars for training
                end = datetime.utcnow()
                start = end - timedelta(days=lookback)
                bars = fetch_bars(
                    token=token,
                    contract_id=contract_id,
                    start_time=start,
                    end_time=end,
                    unit=UNIT,
                    unit_number=UNIT_NUMBER,
                    limit=BAR_LIMIT,
                    live=False
                )
                st.write(f"Fetched {len(bars)} bars for training.")
                df_bars = pd.DataFrame(bars)
                trades = convert_to_trades(df_bars)
                st.write(f"Converted to {len(trades)} trades.")
                X_seq, y = prepare_data(trades)
                st.write(f"Prepared dataset: X_seq shape {X_seq.shape}, y shape {y.shape}.")
                if X_seq.size == 0 or y.size == 0:
                    st.error("No valid training data generated. Try adjusting lookback or checking data.")
                else:
                    # ✅ Iterate the training generator so epochs actually run and are visible
                    progress = st.progress(0)
                    status = st.empty()
                    last_pct = -1
                    for pct in train_model(
                        X_seq, y, input_size=X_seq.shape[2],
                        batch_size=64, num_epochs=60, yield_progress=True
                    ):
                        try:
                            ipct = int(pct)
                        except Exception:
                            ipct = 0
                        if ipct != last_pct:
                            progress.progress(max(0, min(100, ipct)))
                            status.text(f"Training… {ipct}%")
                            last_pct = ipct
                    status.text("Training… 100%")
                    st.success("✅ Model retrained on historical bars.")
                    st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")


if st.button("▶️ Run Simulation"):
    if mode != "sim":
        st.warning("Switch mode to ‘sim’ in the sidebar.")
    else:
        with st.spinner("Running simulation…"):
            try:
                # --- Ensure the scaler is fitted before simulation ---
                from sklearn.preprocessing import StandardScaler
                import joblib
                import numpy as np
                scaler_path = "scaler.pkl"
                if os.path.exists(scaler_path):
                    handler.scaler = joblib.load(scaler_path)
                else:
                    hist_df = load_training_history()
                    if hist_df is not None and not hist_df.empty:
                        feature_cols = [c for c in hist_df.columns if c != "label"]
                        import numpy as np
                        def flatten_row(row):
                            flat = []
                            for val in row:
                                if isinstance(val, (list, tuple, np.ndarray)):
                                    flat.extend(val)
                                else:
                                    flat.append(val)
                            return flat

                        X_cleaned = hist_df[feature_cols].apply(flatten_row, axis=1, result_type='expand')
                        print(f"[INFO] Fitting scaler on data shape: {X_cleaned.shape}")
                        fitted_scaler = StandardScaler().fit(X_cleaned.values)
                        handler.scaler = fitted_scaler
                        joblib.dump(fitted_scaler, scaler_path)
                    else:
                        print("[WARNING] No historical data found. Fitting dummy scaler.")
                        dummy_data = np.zeros((120, 15))  # default shape for fallback
                        handler.scaler = StandardScaler().fit(dummy_data)
                        joblib.dump(handler.scaler, scaler_path)
                # --- End scaler fit logic ---
                handler.run_sim_mode(contract_id)
            except RuntimeError as e:
                st.error(f"Simulation failed: {e}")
                # Don't show success if failed
            else:
                st.success("✅ Simulation complete.")

if st.button("🟢 Start Live"):
    if mode != "live":
        st.warning("Switch mode to ‘live’ in the sidebar.")
    else:
        st.info("Entering live mode. Check logs for trades.")
        handler.run_live_mode(contract_id)
        st.session_state["live_mode"] = True
        st.session_state["live_running"] = True

# Add Stop Live button below the live button
if st.button("🔴 Stop Live"):
    st.session_state["live_mode"] = False
    st.session_state["live_running"] = False
    if hasattr(handler, "kill_live"):
        handler.kill_live()
    st.success("Live mode stopped.")

# If live mode was started, ensure run_live_mode is invoked on every rerun only if not already running
if st.session_state["live_mode"] and not st.session_state["live_running"]:
    st.session_state["live_running"] = True
    handler.run_live_mode(contract_id)


# ── Display results in tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4, = st.tabs([
    "Simulated Trade Log & Training Examples",
    "Pattern Analysis",
    "NN Analysis",
    "📡 Live Monitor",
])

# ---- RAW HISTORICAL BARS & FEATURE MATRIX PREVIEW (Experimental) ----
# Try to find where bars_df is created from historical bars (look for bars_df = pd.DataFrame(bars))
try:
    # Attempt to load bars from a typical file or function, fallback to None if not available
    if os.path.exists("bars.json"):
        with open("bars.json") as f:
            bars = json.load(f)
        bars_df = pd.DataFrame(bars)
        st.subheader("📊 Raw Historical Bars Preview")
        st.dataframe(bars_df.tail(50))
        # Try to generate SMC features if smc_logic is available
        try:
            from smc_logic import smc

            swings = smc.swing_highs_lows(bars_df)
            bos = smc.bos_choch(bars_df, swings)
            ob = smc.ob(bars_df, swings)
            fvg = smc.fvg(bars_df)
            liq = smc.liquidity(bars_df, swings)

            train_df = pd.concat([swings, bos, ob, fvg, liq], axis=1)
            st.subheader("🧠 Feature Matrix Preview")
            if train_df is None or train_df.empty:
                st.warning("⚠️ Feature generation failed or returned empty. Check SMC logic.")
            else:
                st.dataframe(train_df.tail(50))
        except Exception:
            pass
except Exception:
    pass

with tab1:
    # Load training history for use in multiple tabs
    train_df = load_training_history()
    if train_df.empty:
        st.warning("⚠️ Not enough valid training data found. This may happen if the bar data is too short. Try increasing the lookback or checking bar preprocessing.")
    st.header("Simulated Trade Log")
    trade_df = load_trade_log()
    if not trade_df.empty:
        st.dataframe(trade_df)
        st.line_chart(trade_df.set_index("exit_index")["equity"])
    else:
        st.write("No simulation data yet. Run the simulation to see results.")

    st.header("Training Examples")
    if not train_df.empty:
        st.dataframe(train_df)
        # Confirm training files exist
        missing_files = []
        for fname in ["training_history.json", "training_metrics.json", "weight_history.json"]:
            if not os.path.exists(fname):
                missing_files.append(fname)
        if missing_files:
            st.warning(f"Missing training data files: {', '.join(missing_files)}.")

with tab2:
    # Ensure train_df is loaded for this tab as well
    train_df = load_training_history()
    st.header("Pattern Analysis")
    st.write(f"training_history.json exists: {os.path.exists('training_history.json')}")
    st.write(f"train_df shape: {train_df.shape}")
    if not train_df.empty:
        if "prev_HH" in train_df.columns and "prev_LL" in train_df.columns:
            # Derive structure label from previous HH/LL flags
            train_df["structure"] = train_df.apply(
                lambda r: "HH" if r["prev_HH"] else ("LL" if r["prev_LL"] else None),
                axis=1
            )

            # Show index, structure, order_block, and fvg for the first 5 examples (including index)
            sample = train_df.reset_index().rename(columns={"index": "idx"})
            sample = sample[["idx", "structure", "order_block", "fvg"]].head(5)
            st.subheader("Sample Patterns")
            st.table(sample)

            # Prepare numeric feature inputs for predictions
            numeric_cols = ["open","high","low","close","volume","prev_HH","prev_LL","order_block","fvg","liquidity_sweep"]
            numeric_sample = train_df[numeric_cols].head(len(sample))
            # NOTE: load_model_and_predict expects a 120×feature sequence, which we don't have here.
            # So just show a placeholder and avoid misleading errors.
            st.subheader("Model Predictions (preview)")
            try:
                _rows = len(numeric_sample)
            except Exception:
                _rows = 0
            st.info("Skipping per-row predictions here because this view does not contain 120×feature sequences. Use the Simulation or Live tabs to see real NN outputs.")

            # ── Feature Correlation Heatmap ──────────────────────────────────────
            st.subheader("Feature Correlation Heatmap")
            # drop non-numeric column 'structure' before computing correlation
            corr_df = sample.drop(columns=['structure']).corr()
            fig, ax = plt.subplots()
            cax = ax.imshow(corr_df.values, aspect='auto')
            ax.set_xticks(range(len(corr_df.columns))); ax.set_xticklabels(corr_df.columns, rotation=90)
            ax.set_yticks(range(len(corr_df.index))); ax.set_yticklabels(corr_df.index)
            for (i,j), val in np.ndenumerate(corr_df.values):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6)
            fig.colorbar(cax, ax=ax)
            st.pyplot(fig)

            # The following prediction-based visualizations are omitted here due to lack of valid predictions

            # If true labels exist, show confusion matrix
            # (Not shown here as no predictions are made)
        else:
            st.warning("Training data lacks 'prev_HH' and 'prev_LL' columns")
    else:
        st.write("No pattern data yet.")

with tab3:
    st.header("Neural Network Analysis")

    # Add sliders in sidebar for learning_rate and dropout
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.01, step=0.0001, value=0.001, format="%.4f")
    dropout = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.9, step=0.05, value=0.2)

    # Dynamically derive input size from available training features
    hist_df = load_training_history()
    if hist_df is not None and not hist_df.empty:
        feature_cols = [c for c in hist_df.columns if c != "label"]
        default_input_neurons = len(feature_cols)
    else:
        feature_cols = []
        default_input_neurons = 10
    default_hidden1_size = 64

    input_neurons = len(feature_cols) if feature_cols else 10
    hidden1_size = st.slider("Hidden Layer 1 Size", min_value=1, max_value=128, value=default_hidden1_size)
    hidden2_size = st.slider("Hidden Layer 2 Size", min_value=1, max_value=128, value=32)

    model = AdvancedSignalPredictor(
        input_size  = input_neurons,
        hidden1     = hidden1_size,
        hidden2     = hidden2_size,
        output_size = 4,
        dropout     = dropout
    )

    model_config = {"lr": learning_rate, "dropout": dropout}
    st.write("\u2692 Model Config:", model_config)

    col1, col2, col3 = st.columns(3)
    # Collect linear layers in forward order if they exist
    linear_layer_names = [n for n in ["fc1", "fc2", "fc3", "out", "classifier"]
                          if hasattr(model, n) and isinstance(getattr(model, n), torch.nn.Linear)]
    linear_layers = [(n, getattr(model, n)) for n in linear_layer_names]
    first_lin = linear_layers[0][1] if linear_layers else None
    last_lin  = linear_layers[-1][1] if linear_layers else None
    col1.metric("Input Neurons", getattr(first_lin, "in_features", input_neurons))
    col2.metric("Hidden Layers", max(0, len(linear_layers) - 1))
    col3.metric("Output Neurons", getattr(last_lin, "out_features", 4))

    # Dynamic network topology diagram
    layer_labels = []
    for i, (n, layer) in enumerate(linear_layers):
        if i == 0:
            layer_labels.append((f"Input\n{layer.in_features}", 1.3))
        # Treat all but last as hidden
        if i < len(linear_layers) - 1:
            layer_labels.append((f"Hidden{i+1}\n{layer.out_features}", 1.6))
        else:
            layer_labels.append((f"Output\n{layer.out_features}", 1.3))

    nodes = []
    edges = []
    for idx, (lbl, width) in enumerate(layer_labels):
        nodes.append(f'  L{idx} [label="{lbl}", width={width}];')
        if idx > 0:
            edges.append(f"  L{idx-1} -> L{idx} [penwidth=2];")

    dot = "\n".join([
        "digraph G {",
        "  rankdir=LR;",
        "  node [shape=circle, style=filled, color=lightblue, fixedsize=true];",
        *nodes,
        *edges,
        "}",
    ])
    st.subheader("Network Topology")
    st.graphviz_chart(dot, use_container_width=True)

    # ── Model Summary ───────────────────────────────────────────────────────
    st.subheader("Model Summary")
    try:
        def _count_params(m):
            return int(sum(p.numel() for p in m.parameters() if p.requires_grad))
        total_params = _count_params(model)
        st.metric("Trainable Parameters", f"{total_params:,}")
        rows = []
        for name, layer in linear_layers:
            rows.append({
                "Layer": name,
                "Type": layer.__class__.__name__,
                "In": getattr(layer, 'in_features', None),
                "Out": getattr(layer, 'out_features', None),
                "Params": int(sum(p.numel() for p in layer.parameters() if p.requires_grad))
            })
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.write("No linear layers detected.")
    except Exception as e:
        st.warning(f"Failed to build model summary: {e}")

    # ── Advanced NN Visualizations ────────────────────────────────────────
    st.subheader("Training Metrics (if available)")
    try:
        metrics = load_training_history_metrics()  # implement this to load 'loss'/'accuracy' JSON
        if not metrics.empty:
            mdf = metrics.set_index("epoch")
            candidate_cols = [c for c in ["loss","accuracy","val_loss","val_accuracy"] if c in mdf.columns]
            if candidate_cols:
                st.line_chart(mdf[candidate_cols], height=250, use_container_width=True)
            else:
                st.write(f"Metrics available: {list(mdf.columns)}")
        else:
            st.write("No training metrics found.")
    except Exception as e:
        st.warning(f"Failed to load training metrics: {e}")

    # ── Forward Pass Trace ──────────────────────────────────────────────────
    st.subheader("Forward Pass Trace")
    try:
        hist_df = load_training_history()
        if hist_df is not None and not hist_df.empty:
            feature_cols = [c for c in hist_df.columns if c != "label"]
            sample = hist_df[feature_cols].iloc[:8].astype(float).values
            x = torch.tensor(sample, dtype=torch.float32)

            trace = []
            hooks = []
            def _mk(name):
                def _hook(m, inp, out):
                    try:
                        arr = out.detach().cpu().numpy()
                        trace.append({
                            "Layer": name,
                            "Shape": str(list(arr.shape)),
                            "Mean": float(np.mean(arr)),
                            "Std": float(np.std(arr)),
                            "Min": float(np.min(arr)),
                            "Max": float(np.max(arr)),
                        })
                    except Exception:
                        trace.append({"Layer": name, "Shape": "?", "Mean": np.nan, "Std": np.nan, "Min": np.nan, "Max": np.nan})
                return _hook
            for name, layer in linear_layers:
                hooks.append(layer.register_forward_hook(_mk(name)))
            with torch.no_grad():
                _ = model(x)
            for h in hooks:
                try: h.remove()
                except Exception: pass
            if trace:
                st.dataframe(pd.DataFrame(trace))
            else:
                st.write("No activations captured.")
        else:
            st.write("No training data available for a forward trace.")
    except Exception as e:
        st.warning(f"Failed to run forward pass trace: {e}")

    st.subheader("Weight Distributions")
    try:
        if not linear_layers:
            st.write("No linear layers found.")
        else:
            for name, layer in linear_layers:
                w = layer.weight.data.detach().cpu().numpy().flatten()
                st.text(f"{name} weights")
                st.bar_chart(pd.Series(w).value_counts(bins=20).sort_index())
    except Exception as e:
        st.warning(f"Failed to plot weight distributions: {e}")

    st.subheader("Sample Activations")
    try:
        hist_df = load_training_history()
        if hist_df is not None and not hist_df.empty:
            feature_cols = [c for c in hist_df.columns if c != "label"]
            sample_features = hist_df[feature_cols].iloc[:5].values.astype(float).values
            x = torch.tensor(sample_features, dtype=torch.float32)

            # Capture activations with forward hooks
            activations = {}
            hooks = []
            for name, layer in linear_layers:
                def _mk_hook(n):
                    def _hook(module, inp, out):
                        try:
                            activations[n] = out.detach().cpu().numpy()
                        except Exception:
                            activations[n] = np.array([])
                    return _hook
                hooks.append(layer.register_forward_hook(_mk_hook(name)))

            with torch.no_grad():
                _ = model(x)

            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

            if not activations:
                st.write("No activations captured.")
            else:
                for name in activations:
                    st.text(f"{name} activations")
                    arr = activations[name]
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    st.bar_chart(pd.DataFrame(arr))
        else:
            st.write("No training data available to show activations.")
    except Exception as e:
        st.warning(f"Failed to generate sample activations: {e}")

    # ── Gradient Flow ───────────────────────────────────────────────────────
    st.subheader("Gradient Flow")
    try:
        hist_df = load_training_history()
        if hist_df is not None and not hist_df.empty:
            feature_cols = [c for c in hist_df.columns if c != "label"]
            sample = hist_df[feature_cols].iloc[:32].astype(float).values
            x = torch.tensor(sample, dtype=torch.float32)
            # Dummy target: zeros with correct output width
            with torch.no_grad():
                out_probe = model(x[:1])
            out_dim = int(out_probe.shape[-1]) if hasattr(out_probe, 'shape') else 1
            y = torch.zeros((x.shape[0], out_dim), dtype=torch.float32)
            # One backward pass
            for p in model.parameters():
                if p.grad is not None: p.grad.zero_()
            preds = model(x)
            loss = torch.nn.functional.mse_loss(preds, y)
            loss.backward()
            # Aggregate grad norms per linear layer
            grad_rows = []
            for name, layer in linear_layers:
                total = 0.0
                for p in layer.parameters():
                    if p.grad is not None:
                        total += float(p.grad.detach().norm(2).cpu().numpy())
                grad_rows.append({"Layer": name, "L2 Grad Norm": total})
            gdf = pd.DataFrame(grad_rows)
            if not gdf.empty:
                st.bar_chart(gdf.set_index("Layer"))
            else:
                st.write("No gradients computed.")
        else:
            st.write("No training data available for gradient flow.")
    except Exception as e:
        st.warning(f"Failed to compute gradient flow: {e}")

    # --- Brain Scan: Activation Atlas ---
    st.subheader("🧠 Brain Scan: Activation Atlas")

    if st.button("🧬 Generate Activation Atlas"):
        try:
            if hist_df is not None and not hist_df.empty and feature_cols:
                sample_inputs = hist_df[feature_cols].sample(min(100, len(hist_df))).values
                atlas_df = build_activation_atlas(model, sample_inputs, layer_name="fc1", n_clusters=8)
                if not atlas_df.empty:
                    st.vega_lite_chart(atlas_df, {
                        "mark": "circle",
                        "encoding": {
                            "x": {"field": "x", "type": "quantitative"},
                            "y": {"field": "y", "type": "quantitative"},
                            "color": {"field": "cluster", "type": "nominal"}
                        }
                    })
                else:
                    st.write("No atlas data produced.")
            else:
                st.write("No training data available for atlas.")
        except Exception as e:
            st.warning(f"Failed to generate activation atlas: {e}")

    st.subheader("🔥 Neuron Activation Timeline")
    if st.button("📈 Show Activation Timeline"):
        try:
            if hist_df is not None and not hist_df.empty and feature_cols:
                sample_inputs = hist_df[feature_cols].values[:200]
                timeline_fig = activation_timeline(model, sample_inputs, layer_name="fc1")
                st.pyplot(timeline_fig)
            else:
                st.write("No training data available for timeline.")
        except Exception as e:
            st.warning(f"Failed to generate timeline: {e}")

    st.subheader("🧪 Feature Maximization")
    if hasattr(model, 'fc1') and isinstance(model.fc1, torch.nn.Linear):
        neuron_idx = st.slider("Neuron Index", 0, int(model.fc1.out_features) - 1, 0)
        if st.button("🎯 Maximize Neuron"):
            try:
                maximized_input = feature_maximization(model, "fc1", neuron_idx, input_dim=len(feature_cols) or input_neurons)
                if feature_cols:
                    st.bar_chart(pd.Series(maximized_input.flatten(), index=feature_cols))
                else:
                    st.bar_chart(pd.Series(maximized_input.flatten()))
            except Exception as e:
                st.warning(f"Feature maximization failed: {e}")
    else:
        st.write("fc1 layer not available for maximization.")

    st.subheader("🔍 Feature Influence (Saliency Approx.)")
    try:
        import numpy as np
        hist_df = load_training_history()
        if hist_df is not None and not hist_df.empty and 'label' in hist_df.columns:
            sample_features = hist_df.drop("label", axis=1).iloc[0].tolist()
        elif hist_df is not None and not hist_df.empty:
            sample_features = hist_df.iloc[0].tolist()
        else:
            sample_features = None
        if sample_features is None:
            st.write("No data for saliency map.")
        else:
            sample_input = torch.tensor([sample_features], dtype=torch.float32).requires_grad_()
            output = model(sample_input)
            output.grad = None
            model.zero_grad()
            scalar_output = output[0].sum()
            scalar_output.backward()
            saliency = sample_input.grad.abs().detach().numpy().flatten()
            if hist_df is not None and not hist_df.empty and 'label' in hist_df.columns:
                feature_names = hist_df.drop("label", axis=1).columns.tolist()
            else:
                feature_names = [f"f{i}" for i in range(len(saliency))]
            sal_df = pd.DataFrame({"Feature": feature_names, "Importance": saliency})
            fig, ax = plt.subplots()
            s = sal_df.sort_values("Importance", ascending=False)
            ax.barh(s["Feature"], s["Importance"]) 
            ax.invert_yaxis()
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Failed to generate saliency map: {e}")

    st.subheader(" Latest Model Prediction")
    try:
        hist_df = load_training_history()
        feature_cols = [c for c in hist_df.columns if c != "label"] if hist_df is not None and not hist_df.empty else feature_cols
        latest_tick = st.session_state.get("latest_tick")
        if latest_tick:
            features = [latest_tick.get(col, 0) for col in feature_cols]
            x = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                out = model(x).squeeze().numpy()
            st.write({"take_trade": round(float(out[0]), 2), "confidence": round(float(out[1]), 2), "tp_mult": round(float(out[2]), 2), "sl_mult": round(float(out[3]), 2)})
        else:
            st.write("No recent tick data to analyze.")
    except Exception as e:
        st.warning(f"Failed to generate latest model prediction: {e}")

    # ── Inference Latency (CPU) ─────────────────────────────────────────────
    st.subheader("Inference Latency (CPU)")
    try:
        hist_df = load_training_history()
        if hist_df is not None and not hist_df.empty:
            feature_cols = [c for c in hist_df.columns if c != "label"]
            sample = hist_df[feature_cols].iloc[:64].astype(float).values
            x = torch.tensor(sample, dtype=torch.float32)
            # Warmup
            with torch.no_grad():
                _ = model(x[:8])
            import time
            iters = 20
            start = time.time()
            with torch.no_grad():
                for _ in range(iters):
                    _ = model(x)
            dur = time.time() - start
            per_batch_ms = 1000.0 * dur / iters
            per_sample_ms = per_batch_ms / len(x)
            colA, colB = st.columns(2)
            colA.metric("ms / batch", f"{per_batch_ms:.2f}")
            colB.metric("ms / sample", f"{per_sample_ms:.3f}")
        else:
            st.write("No data available to time inference.")
    except Exception as e:
        st.warning(f"Failed to measure latency: {e}")

    st.subheader(" Weight Drift Tracker")
    try:
        if os.path.exists("weight_history.json"):
            with open("weight_history.json") as f:
                w_hist = json.load(f)
            df = pd.DataFrame(w_hist)
            # Only plot if there are numeric columns and df is not empty
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            if df.empty or not any(numeric_cols):
                st.write("Weight history file is empty or no numeric columns.")
            else:
                if df.index.dtype != 'int64':
                    df.index = range(len(df))
                st.line_chart(df[numeric_cols])
        else:
            st.write("No historical weight data available.")
    except Exception as e:
        st.warning(f"Failed to load weight history: {e}")

    st.subheader("❌ Misclassification Analyzer")
    try:
        train_df = load_training_history()
        if train_df is not None and not train_df.empty:
            if 'label' in train_df.columns:
                features_df = train_df.drop("label", axis=1)
                true_labels = train_df["label"]
            else:
                features_df = train_df
                true_labels = None

            if not features_df.empty:
                preds = [load_model_and_predict(row.tolist()) for row in features_df.values]
                pred_labels = pd.Series([p[0] for p in preds])
                if true_labels is not None:
                    mismatches = train_df[true_labels != pred_labels]
                    if not mismatches.empty:
                        st.write("Misclassified Examples:")
                        st.dataframe(mismatches)
                    else:
                        st.write("No misclassifications found.")
                else:
                    st.write("No true labels to compare against.")
            else:
                st.write("No feature data available for misclassification analysis.")
        else:
            st.write("No training data available for misclassification analysis.")
    except Exception as e:
        st.warning(f"Failed misclassification analysis: {e}")


# ── Live Monitor Tab ──────────────────────────────────────────────────────
with tab4:
    # Placeholders for live monitor
    account_placeholder = st.empty()
    orders_placeholder = st.empty()
    positions_placeholder = st.empty()
    tick_chart_placeholder = st.empty()
    equity_chart_placeholder = st.empty()

# ── Live Monitor Dynamic Update ─────────────────────────────────────────────
if st.session_state["live_mode"]:
    # Single-pass update, then schedule a rerun to avoid blocking the app.
    refresh_secs = st.sidebar.slider("Live refresh (sec)", 0.5, 5.0, 1.0, 0.5)

    account_placeholder.subheader("Live Account Info")
    account_placeholder.json(st.session_state.get("account_info", {}))
    orders_placeholder.subheader("Live Orders")
    orders_placeholder.json(st.session_state.get("orders", []))
    positions_placeholder.subheader("Live Positions")
    positions_placeholder.json(st.session_state.get("positions", []))

    tick_data = st.session_state.get("tick_data", [])
    if tick_data:
        tick_df = pd.DataFrame(tick_data)
        # Altair chart for tick data with explicit type for "interval" if present
        if "interval" in tick_df.columns:
            import altair as alt
            tick_chart_placeholder.altair_chart(
                alt.Chart(tick_df).mark_line().encode(
                    x=alt.X("interval:N", title="Interval"),
                    y="price"
                ),
                use_container_width=True
            )
        else:
            # Guard against missing timestamp
            if "timestamp" in tick_df.columns:
                tick_chart_placeholder.line_chart(tick_df.set_index("timestamp")["price"])
            else:
                tick_chart_placeholder.line_chart(tick_df["price"])
    else:
        tick_chart_placeholder.write("No live ticks received yet.")

    if os.path.exists("live_equity.json"):
        with open("live_equity.json") as f:
            eq_df = pd.DataFrame(json.load(f))
        if not eq_df.empty and "timestamp" in eq_df.columns and "equity" in eq_df.columns:
            equity_chart_placeholder.line_chart(eq_df.set_index("timestamp")["equity"])
        else:
            equity_chart_placeholder.write("No equity curve data yet.")
    else:
        equity_chart_placeholder.write("No equity curve data yet.")

    time.sleep(refresh_secs)
    st.rerun()

    with st.expander("What you're seeing"):
        st.markdown(
            """
            **Model Summary** shows parameter counts per layer.
            **Forward Pass Trace** logs shapes and value stats after each linear layer.
            **Sample Activations** displays raw activations for a few samples.
            **Gradient Flow** backprops one synthetic step to surface vanishing/exploding risks.
            **Inference Latency** provides CPU timing to benchmark changes.
            """
        )