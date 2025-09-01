from datetime import timedelta
import time
import logging
import json
import threading
from datetime import datetime
from collections import deque
import pandas as pd
import requests
from signalrcore.hub_connection_builder import HubConnectionBuilder

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Utility to get the latest live contract id for a given search_text (default "ES")
def get_latest_contract_id(token, search_text="ES"):
    logger.info(f"Getting latest contract id with token: {token[:5]}... and search_text: {search_text}")
    contracts = search_contracts(token, search_text, live=True)
    if contracts:
        return contracts[0]["contractId"]
    else:
        logger.warning("No live contracts found, falling back to non-live contracts")
        contracts = search_contracts(token, search_text, live=False)
        if contracts:
            return contracts[0]["contractId"]
        else:
            logger.error("No contracts found at all")
            raise Exception("No contracts found")

# LiveCandleAggregator aggregates trade data into minute candles
class LiveCandleAggregator:
    def __init__(self):
        self.current_bar = None
        self.bars = deque(maxlen=1000)

    def update(self, trade):
        logger.debug(f"[LiveCandleAggregator] Incoming message type: {type(trade)}")
        logger.debug(f"[LiveCandleAggregator] Raw message content: {trade}")
        if isinstance(trade, str):
            try:
                trade = json.loads(trade)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse trade JSON: {e} | Raw data: {trade}")
                return
        if not isinstance(trade, dict):
            logger.warning(f"Skipping non-dict tick: {trade}")
            return
        ts_raw = trade.get("timestamp") or trade.get("t")
        price_raw = trade.get("price") or trade.get("p")
        volume_raw = trade.get("size") or trade.get("q") or trade.get("volume")

        if ts_raw is None or price_raw is None or volume_raw is None:
            logger.warning(f"Skipping malformed tick: {trade}")
            return

        ts = pd.to_datetime(ts_raw)
        price = float(price_raw)
        try:
            volume = int(volume_raw)
        except (TypeError, ValueError):
            logger.warning(f"Skipping trade due to invalid volume: {trade}")
            return
        minute = ts.floor("min")

        if self.current_bar is None or self.current_bar["time"] != minute:
            if self.current_bar:
                self.bars.append(self.current_bar)
            self.current_bar = {
                "time": minute,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume
            }
        else:
            self.current_bar["high"] = max(self.current_bar["high"], price)
            self.current_bar["low"] = min(self.current_bar["low"], price)
            self.current_bar["close"] = price
            self.current_bar["volume"] += volume

    def get_dataframe(self):
        df = pd.DataFrame(self.bars)
        if self.current_bar:
            df = pd.concat([df, pd.DataFrame([self.current_bar])], ignore_index=True)
        return df


# Partial close a position for a given account and contract
def partial_close_position(token, account_id, contract_id, size):
    url = "https://api.topstepx.com/api/Position/partialCloseContract"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "accountId": account_id,
        "contractId": contract_id,
        "size": size
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            return True
        else:
            raise Exception(f"Partial close failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Partial close request failed with status: {response.status_code}")

def search_contract_by_id(token, contract_id):
    url = "https://api.topstepx.com/api/Contract/searchById"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "contractId": contract_id
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success") and "contracts" in result:
            return result["contracts"]
        else:
            raise Exception(f"Contract search by ID failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Contract search by ID request failed with status: {response.status_code}")
def search_contracts(token, search_text, live=False):
    url = "https://api.topstepx.com/api/Contract/search"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "searchText": search_text,
        "live": live
    }

    logger.info(f"🔐 Token: {token[:5]}... 🔍 Search text: {search_text}")
    logger.info(f"📦 Payload: {data}")
    logger.info(f"📨 URL: {url}")
    logger.info(f"📨 Headers: {headers}")

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success") and "contracts" in result:
            return result["contracts"]
        else:
            logger.error(f"Contract search failed: {result.get('errorMessage')}")
            raise Exception(f"Contract search failed: {result.get('errorMessage')}")
    else:
        logger.error(f"Contract search request failed with status: {response.status_code}")
        raise Exception(f"Contract search request failed with status: {response.status_code}")

MARKET_HUB_URL = "https://rtc.topstepx.com/hubs/market"
USER_HUB_URL = "https://rtc.topstepx.com/hubs/user"

def login_and_get_token(username, api_key):
    url = "https://api.topstepx.com/api/Auth/loginKey"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json"
    }
    data = {
        "userName": username,
        "apiKey": api_key
    }

    response = requests.post(url, headers=headers, json=data)
    logger.info(f"[LOGIN] Status: {response.status_code}")
    logger.info(f"[LOGIN] Body: {response.text}")
    if response.status_code == 200:
        try:
            result = response.json()
        except Exception as e:
            logger.error(f"[LOGIN] Failed to parse JSON: {e}")
            raise Exception(f"[LOGIN] Failed to parse JSON: {e}")

        if result.get("success") and result.get("token"):
            token = result["token"]
            print("[DEBUG] TOKEN:", token)
            return token
        else:
            logger.error(f"[LOGIN] Auth failed: {result}")
            raise Exception(f"[LOGIN] Auth failed: {result}")
    else:
        logger.error(f"[LOGIN] Auth request failed with status: {response.status_code}")
        raise Exception(f"[LOGIN] Auth request failed with status: {response.status_code}")

def validate_token(token):
    url = "https://api.topstepx.com/api/Auth/validate"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        result = response.json()
        if result.get("success") and result.get("newToken"):
            return result["newToken"]
        else:
            raise Exception(f"Token validation failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Token validation request failed with status: {response.status_code}")

def get_active_accounts(token):
    url = "https://api.topstepx.com/api/Account/search"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "onlyActiveAccounts": True
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success") and "accounts" in result:
            return result["accounts"]
        else:
            raise Exception(f"Account search failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Account search request failed with status: {response.status_code}")


# --- WebSocket-based SignalRClient implementation ---
import json
import threading
import websocket
import logging

class SignalRClient:
    """
    Direct WebSocket client for TopstepX real-time API.
    Replaces signalrcore-based client with lightweight raw WebSocket.
    """

    def __init__(self, token, contract_id=None, hub_type="market"):
        if hub_type not in ("market", "user"):
            raise ValueError("hub_type must be 'market' or 'user'")

        self.url = f"wss://rtc.topstepx.com/hubs/{hub_type}?access_token={token}"
        self.contract_id = contract_id
        self.ws = None
        self.tick_listeners = []
        self._stop = False
        self.connected = False

    def connect(self):
        def on_message(ws, message):
            messages = message.split(chr(30))
            for msg in messages:
                if not msg.strip():
                    continue
                try:
                    data = json.loads(msg)
                    # Handle GatewayTrade messages containing trade data
                    if data.get("target") == "GatewayTrade":
                        arguments = data.get("arguments", [])
                        if isinstance(arguments, list) and len(arguments) == 2:
                            contract_id, trade_data_list = arguments
                            if isinstance(trade_data_list, list):
                                for trade_data in trade_data_list:
                                    for listener in self.tick_listeners:
                                        listener(trade_data)
                            else:
                                # Single trade dict (fallback)
                                for listener in self.tick_listeners:
                                    listener(trade_data_list)
                        else:
                            logging.error(f"[WebSocket] Unexpected GatewayTrade arguments structure: {arguments}")
                    else:
                        for listener in self.tick_listeners:
                            listener(data)
                except Exception as e:
                    logging.error(f"[WebSocket] Message processing failed: {e} | Raw: {msg}")

        def on_error(ws, error):
            logging.error(f"[WebSocket] Error: {error}")

        def on_close(ws, close_status_code, close_msg):
            self.connected = False
            logging.warning("[WebSocket] Connection closed.")

        def on_open(ws):
            self.connected = True
            logging.info("[WebSocket] Connection opened.")
            # Send SignalR handshake
            handshake_msg = '{"protocol":"json","version":1}' + chr(30)
            self.ws.send(handshake_msg)
            logging.info("[WebSocket] Sent SignalR handshake.")
            if self.contract_id:
                self.subscribe_to_market(self.contract_id)

        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def subscribe_to_market(self, contract_id):
        self.send([contract_id], target="SubscribeContractTrades")

    def unsubscribe_from_market(self, contract_id):
        self.send([contract_id], target="UnsubscribeContractTrades")

    def subscribe_user_updates(self, account_id):
        self.send([], target="SubscribeAccounts")
        self.send([account_id], target="SubscribeOrders")
        self.send([account_id], target="SubscribePositions")
        self.send([account_id], target="SubscribeTrades")

    def unsubscribe_user_updates(self, account_id):
        self.send([], target="UnsubscribeAccounts")
        self.send([account_id], target="UnsubscribeOrders")
        self.send([account_id], target="UnsubscribePositions")
        self.send([account_id], target="UnsubscribeTrades")

    def send(self, arguments, target="UnknownMethod"):
        if self.ws and self.connected:
            try:
                payload = {
                    "arguments": arguments,
                    "target": target,
                    "type": 1
                }
                formatted_msg = json.dumps(payload) + chr(30)
                self.ws.send(formatted_msg)
                logging.info(f"[WebSocket] Sent: {formatted_msg}")
            except Exception as e:
                logging.error(f"[WebSocket] Failed to send message: {e}")
        else:
            logging.warning("[WebSocket] Cannot send, socket not connected.")

    def on_tick(self, callback):
        self.tick_listeners.append(callback)

    def add_tick_listener(self, listener):
        """
        Register an external tick listener callback.
        """
        self.tick_listeners.append(listener)

    def close(self):
        self._stop = True
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logging.error(f"[WebSocket] Error closing connection: {e}")

def get_historical_bars(token, contract_id, start_time, end_time, unit=3, unit_number=1, limit=100, live=False, include_partial_bar=False):
    url = "https://api.topstepx.com/api/History/retrieveBars"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "contractId": contract_id,
        "live": live,
        "startTime": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unit": unit,
        "unitNumber": unit_number,
        "limit": limit,
        "includePartialBar": include_partial_bar
    }

    logger.info(f"📦 Historical payload: {json.dumps(data, indent=2)}")
    response = requests.post(url, headers=headers, json=data)
    logger.info(f"📨 Response status: {response.status_code}")
    logger.info(f"📨 Response body: {response.text}")

    # Check response is not None
    if response is None:
        raise Exception("❌ No response received from historical bars endpoint.")

    if response.status_code == 200:
        try:
            result = response.json()
        except Exception as e:
            raise Exception(f"Failed to parse response JSON: {e}")

        if result is None:
            raise Exception("❌ Received null response from historical bars endpoint.")

        if result.get("success") and "bars" in result and isinstance(result["bars"], list):
            return [{
                "timestamp": bar["t"],
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": bar["v"]
            } for bar in result["bars"]]
        else:
            raise Exception("❌ 'bars' field is missing or not a list in historical bars response.")
    else:
        raise Exception(f"Historical bars request failed with status: {response.status_code}")


# Fetch months of bars by walking backwards in time in chunks
def search_orders(token, account_id, start_timestamp, end_timestamp=None):
    url = "https://api.topstepx.com/api/Order/search"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "accountId": account_id,
        "startTimestamp": start_timestamp
    }
    if end_timestamp:
        data["endTimestamp"] = end_timestamp

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success") and "orders" in result:
            return result["orders"]
        else:
            raise Exception(f"Order search failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Order search request failed with status: {response.status_code}")
def search_open_orders(token, account_id):
    url = "https://api.topstepx.com/api/Order/searchOpen"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "accountId": account_id
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success") and "orders" in result:
            return result["orders"]
        else:
            raise Exception(f"Open order search failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Open order search request failed with status: {response.status_code}")
def place_order(token, account_id, contract_id, order_type, side, size,
                limit_price=None, stop_price=None, trail_price=None,
                custom_tag=None, linked_order_id=None):
    url = "https://api.topstepx.com/api/Order/place"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "accountId": account_id,
        "contractId": contract_id,
        "type": order_type,
        "side": side,
        "size": size,
        "limitPrice": limit_price,
        "stopPrice": stop_price,
        "trailPrice": trail_price,
        "customTag": custom_tag,
        "linkedOrderId": linked_order_id
    }
    # Remove keys with None values (API expects only provided fields)
    data = {k: v for k, v in data.items() if v is not None}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success") and "orderId" in result:
            return result["orderId"]
        else:
            raise Exception(f"Order placement failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Order placement request failed with status: {response.status_code}")
def cancel_order(token, account_id, order_id):
    url = "https://api.topstepx.com/api/Order/cancel"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "accountId": account_id,
        "orderId": order_id
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            return True
        else:
            raise Exception(f"Order cancel failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Order cancel request failed with status: {response.status_code}")
def modify_order(token, account_id, order_id, size=None, limit_price=None, stop_price=None, trail_price=None):
    url = "https://api.topstepx.com/api/Order/modify"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "accountId": account_id,
        "orderId": order_id,
        "size": size,
        "limitPrice": limit_price,
        "stopPrice": stop_price,
        "trailPrice": trail_price
    }
    # Remove keys with None values
    data = {k: v for k, v in data.items() if v is not None}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            return True
        else:
            raise Exception(f"Order modify failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Order modify request failed with status: {response.status_code}")
def close_position(token, account_id, contract_id):
    url = "https://api.topstepx.com/api/Position/closeContract"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "accountId": account_id,
        "contractId": contract_id
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            return True
        else:
            raise Exception(f"Close position failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Close position request failed with status: {response.status_code}")
def search_open_positions(token, account_id):
    url = "https://api.topstepx.com/api/Position/searchOpen"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "accountId": account_id
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success") and "positions" in result:
            return result["positions"]
        else:
            raise Exception(f"Position search failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Position search request failed with status: {response.status_code}")
def search_trades(token, account_id, start_timestamp, end_timestamp=None):
    url = "https://api.topstepx.com/api/Trade/search"
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "accountId": account_id,
        "startTimestamp": start_timestamp
    }
    if end_timestamp:
        data["endTimestamp"] = end_timestamp

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success") and "trades" in result:
            return result["trades"]
        else:
            raise Exception(f"Trade search failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Trade search request failed with status: {response.status_code}")


# Fetch equity and PnL data for an account
def get_equity_and_pnl(token, account_id):
    url = "https://api.topstepx.com/api/Account/get"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "accountId": account_id
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if result.get("success") and "account" in result:
            account = result["account"]
            return {
                "equity": account.get("equity"),
                "balance": account.get("balance"),
                "realized_pnl": account.get("realizedPnl"),
                "unrealized_pnl": account.get("unrealizedPnl")
            }
        else:
            raise Exception(f"Equity/PnL fetch failed: {result.get('errorMessage')}")
    else:
        raise Exception(f"Equity/PnL request failed with status: {response.status_code}")
# Fetch months of bars by walking backwards in time in chunks
def fetch_bars(token, contract_id, start_time, end_time, unit, unit_number, limit=1500, live=False):
    all_bars = []
    current_end = end_time
    while True:
        try:
            raw_bars = get_historical_bars(
                token,
                contract_id,
                start_time,
                current_end,
                unit,
                unit_number,
                limit,
                live=live
            )
            if isinstance(raw_bars, dict) and raw_bars.get("errorCode") == 401:
                raise Exception("401 Token Expired")
        except Exception as e:
            print(f"[DEBUG] Error fetching bars: {e}. Retrying in 5 seconds...")
            if hasattr(e, 'response') and e.response is not None:
                print(f"[DEBUG] Response content: {e.response.text}")
            time.sleep(5)
            continue
        print(f"[DEBUG] raw_bars returned: {raw_bars}")
        # normalize timestamp field
        bars = []
        for b in raw_bars:
            ts = b.get("t") or b.get("timestamp") or b.get("startTime") or b.get("time")
            if ts is None:
                continue
            bars.append({
                "timestamp": datetime.fromisoformat(ts.replace("Z", "+00:00")),
                "open": b.get("o") or b.get("open") or b.get("openPrice"),
                "high": b.get("h") or b.get("high") or b.get("highPrice"),
                "low":  b.get("l") or b.get("low") or b.get("lowPrice"),
                "close":b.get("c") or b.get("close") or b.get("closePrice"),
                "volume":b.get("v") or b.get("volume")
            })
        print(f"[DEBUG] normalized bars count: {len(bars)}")
        if not bars:
            break
        all_bars.extend(bars)
        # find oldest timestamp in this batch
        oldest_ts = min(b['timestamp'] for b in bars)
        oldest_time = oldest_ts
        if oldest_time <= start_time.replace(tzinfo=oldest_time.tzinfo):
            break
        # set next end one microsecond before oldest
        current_end = oldest_time - timedelta(microseconds=1)
    # Debug print before clipping to start_time
    print(f"[DEBUG] Returning {len(all_bars)} bars before clipping to start_time")
    # clip to start_time, ensure start_time is offset-aware to match timestamp
    return [b for b in all_bars if b["timestamp"] >= start_time.replace(tzinfo=b["timestamp"].tzinfo)]