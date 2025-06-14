# main.py (FastAPI Backend - Corrected and Merged)

from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
import random
import asyncio
import jwt
from passlib.context import CryptContext
import httpx  # For making API calls

from firebase_admin import credentials, auth as firebase_admin_auth

from dotenv import load_dotenv
load_dotenv() # This should be one of the first lines

import os
import firebase_admin
from firebase_admin import credentials
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm # Ensure OAuth2PasswordBearer is imported


# Define oauth2_scheme globally so it can be used by dependency functions
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrl is for OpenAPI docs, can be a dummy if not using FastAPI's own token endpoint

try:
    cred_path = os.getenv("FIREBASE_ADMIN_SDK_JSON_PATH")
    if cred_path and os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
    else:
        print("WARNING: FIREBASE_ADMIN_SDK_JSON_PATH not set or file not found. Firebase Admin dependent features will fail.")
except Exception as e:
    print(f"ERROR initializing Firebase Admin SDK: {e}")
load_dotenv()

# --- Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY", "a_very_strong_default_secret_key_for_dev_only_replace_me")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY") # Added
COINMARKETCAP_BASE_URL = "https://pro-api.coinmarketcap.com" # Use "https://sandbox-api.coinmarketcap.com" for testing
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# main.py - in your helper functions section

async def get_current_firebase_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    # Here, oauth2_scheme is re-used, but the token it extracts is now a Firebase ID token
    if not firebase_admin._DEFAULT_APP_NAME in firebase_admin._apps:  # Check if admin app is initialized
        raise HTTPException(status_code=500, detail="Firebase Admin not initialized on server.")
    try:
        # The token comes in as "Bearer <ID_TOKEN>", so strip "Bearer "
        id_token = token  # Assuming token is already stripped of "Bearer " by frontend or a prior step
        if token.lower().startswith("bearer "):
            id_token = token.split(" ", 1)[1]

        decoded_token = firebase_admin_auth.verify_id_token(id_token)
        # You now have decoded_token['uid'], decoded_token['email'], etc.
        # You might want to fetch/create your application-specific user from your DB using uid or email.

        # For this example, we'll just return the decoded token.
        # In a real app, you'd look up your internal user based on decoded_token['uid']
        # and return your UserInDBBase object.

        # Simplified: fetch our internal user based on email from decoded token
        app_user_dict = fake_users_db.get(decoded_token['email'])
        if not app_user_dict:
            # This case might mean a Firebase user exists but not yet in our app's DB.
            # Handle this: e.g., create a local user record now, or raise an error.
            # For now, we'll treat it as an issue if they are not in our DB.
            # This logic should align with what happens during registration.
            print(f"Firebase user {decoded_token['email']} not found in local DB.")
            # If auto-creation of local user on first Firebase auth is desired:
            # portfolio_id = f"{decoded_token['email'].replace('@', '_').replace('.', '_')}_portfolio"
            # app_user_dict = { "email": decoded_token['email'], "full_name": decoded_token.get('name'), "hashed_password": "FIREBASE_AUTH", "disabled": False, "portfolio_id": portfolio_id, "alerts": [], "preferences": UserPreferences().model_dump() }
            # fake_users_db[decoded_token['email']] = app_user_data
            # fake_portfolios_db[portfolio_id] = {"summary": {...}, "assets":[]}
            # For now, let's assume the user should exist if token is valid.
            raise HTTPException(status_code=404, detail="User profile not found in application database.")

        # Convert to UserInDBBase before returning, if you need methods/validation from it
        # For this example, returning the dict for simplicity.
        # A better approach would be to return your UserInDBBase model instance
        # user_obj = UserInDBBase(**app_user_dict)
        # user_obj.uid = decoded_token['uid'] # Add Firebase UID if needed by endpoints
        # return user_obj

        # Add Firebase UID to the app_user_dict so endpoints can use it
        app_user_dict['uid'] = decoded_token['uid']
        return app_user_dict  # Return the dict for now

    except firebase_admin.auth.InvalidIdTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid Firebase ID token: {e}")
    except Exception as e:
        print(f"Error verifying Firebase token: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not verify Firebase credentials")
def verify_password(plain_password, hashed_password): return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password): return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy();
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES));
    to_encode.update({"exp": expire, "iat": datetime.utcnow()});
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


# --- Pydantic Models (Ensure all are defined here) ---
class UserPreferences(BaseModel): theme: str = "theme-dark"
class TokenData(BaseModel): email: Optional[str] = None
class UserInDBBase(BaseModel): email: EmailStr; full_name: Optional[str] = None; hashed_password: str; disabled: Optional[bool] = False; portfolio_id: str; alerts: List[Dict[str, Any]] = Field(default_factory=list); preferences: UserPreferences = Field(default_factory=UserPreferences);
class Config: from_attributes = True
class User(BaseModel): email: EmailStr; full_name: Optional[str] = None; disabled: Optional[bool] = None; preferences: UserPreferences = Field(default_factory=UserPreferences);
class Config: from_attributes = True
class UserCreate(BaseModel): email: EmailStr; full_name: Optional[str] = None; password: str
class Token(BaseModel): access_token: str; token_type: str; user_name: Optional[str] = None
class PredictionRequest(BaseModel): model_type: str; time_horizon: str
class PredictionResponse(BaseModel): asset_id: str; prediction: Dict[str, Any]; confidence_interval: Optional[Dict[str, float]] = None
class TechnicalIndicators(BaseModel): MACD: Optional[Dict[str, Optional[float]]] = None; RSI: Optional[float] = None; BollingerBands: Optional[Dict[str, float]] = None; MovingAverages: Optional[Dict[str, List[Optional[float]]]] = None
class SentimentData(BaseModel): score: float; label: str; sources: str; details: Optional[str] = None
class OnChainMetrics(BaseModel): NVT_ratio: Optional[float] = None; active_addresses: Optional[int] = None
class PortfolioAssetInDB(BaseModel): id: str; asset_id: str; type: str; quantity: float; average_purchase_price: float; current_price: float; current_value: float; pnl: float; pnl_percent: float
class PortfolioSummary(BaseModel): total_value: float; total_pnl: float; total_pnl_percent: float; sharpe_ratio: Optional[float] = None
class PortfolioResponse(BaseModel): summary: PortfolioSummary; assets: List[PortfolioAssetInDB]
class AssetPortfolioCreate(BaseModel): asset_id: str = Field(..., examples=["bitcoin", "EUR/USD"]); type: str = Field(..., examples=["crypto", "forex"]); quantity: float = Field(..., gt=0); purchase_price: float = Field(..., gt=0)
class AlertCreate(BaseModel): asset_id: str; condition_type: str; target_value: float; notification_methods: List[str]
class AlertResponse(AlertCreate): id: str; user_id: str; status: str = "active"; created_at: datetime = Field(default_factory=datetime.utcnow)
class BacktestRequest(BaseModel): strategy_id: str; asset_id: str; start_date: str; end_date: str; initial_capital: float = 10000; leverage: float = 1.0
class EquityPoint(BaseModel): timestamp: str; value: float
class BacktestResult(BaseModel): strategy_id: str; asset_id: str; total_return: float; sharpe_ratio: float; max_drawdown: float; num_trades: int; details: Dict[str, Any]
class ChatQuery(BaseModel): query: str
class ChatReply(BaseModel): reply: str
class ConsultationBookingRequest(BaseModel): consultation_type: str; scheduled_time: datetime; notes: Optional[str] = None; payment_method: str
class ConsultationBookingResponse(BaseModel): booking_id: str; status: str; message: str; payment_url: Optional[str] = None; client_secret: Optional[str] = None; payment_id: Optional[str] = None
class QuantumPortfolioInput(BaseModel): assets: List[Dict[str, Any]]; risk_tolerance: str
class AutoMLStrategyParams(BaseModel): target_metric: str; feature_set: str; asset_id: str
class AltDataRequest(BaseModel): sources: List[str]
class MarketNewsArticle(BaseModel): title: str; description: Optional[str] = None; url: str; source: str; published_at: datetime
class MarketNewsResponse(BaseModel): articles: List[MarketNewsArticle]


# --- Mock Database as before ---
fake_users_db = { "user@example.com": { "email": "user@example.com", "full_name": "Test User", "hashed_password": get_password_hash("string"), "disabled": False, "portfolio_id": "user_example_com_portfolio", "alerts": [], "preferences": UserPreferences().model_dump() } }
fake_portfolios_db = { "user_example_com_portfolio": { "summary": {"total_value": 12500.00, "total_pnl": 500.00, "total_pnl_percent": 0.04, "sharpe_ratio": 1.2}, "assets": [ {"id": "btc1", "asset_id": "BTC/USD", "type":"crypto", "quantity": 0.1, "average_purchase_price": 50000, "current_price": 52000, "current_value": 5200, "pnl": 200, "pnl_percent": 0.04}, {"id": "eth1", "asset_id": "ETH/USD", "type":"crypto", "quantity": 2, "average_purchase_price": 3500, "current_price": 3650, "current_value": 7300, "pnl": 300, "pnl_percent": 0.042}, ] } }


# --- Helper Functions ---
async def get_current_user_from_db(email: str) -> Optional[UserInDBBase]: user_dict = fake_users_db.get(email); return UserInDBBase(**user_dict) if user_dict else None

async def get_current_active_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try: payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]); email: Optional[str] = payload.get("sub");
    except jwt.ExpiredSignatureError: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired", headers={"WWW-Authenticate": "Bearer"})
    except jwt.PyJWTError: raise credentials_exception
    if email is None: raise credentials_exception
    user = await get_current_user_from_db(email);
    if user is None or user.disabled: raise credentials_exception
    return user

def _recalculate_portfolio_summary(portfolio_id: str):
    if portfolio_id not in fake_portfolios_db or "assets" not in fake_portfolios_db[portfolio_id]:
        if portfolio_id not in fake_portfolios_db:
            fake_portfolios_db[portfolio_id] = {"summary": {}, "assets": []}
        elif "assets" not in fake_portfolios_db[portfolio_id]:
            fake_portfolios_db[portfolio_id]["assets"] = []
        if "summary" not in fake_portfolios_db[portfolio_id]:
             fake_portfolios_db[portfolio_id]["summary"] = {}
        fake_portfolios_db[portfolio_id]['summary'].update({'total_value': 0.0, 'total_pnl': 0.0, 'total_pnl_percent': 0.0, 'sharpe_ratio': 0.0})
        return

    portfolio = fake_portfolios_db[portfolio_id]
    total_value = sum(asset.get('current_value', 0.0) for asset in portfolio['assets'])
    total_initial_investment = sum(asset.get('average_purchase_price', 0.0) * asset.get('quantity', 0.0) for asset in portfolio['assets'])
    total_pnl = total_value - total_initial_investment
    total_pnl_percent = (total_pnl / total_initial_investment) if total_initial_investment > 0.000001 else 0.0
    if "summary" not in portfolio: portfolio["summary"] = {} # Ensure summary dict exists
    portfolio['summary'].update({'total_value': total_value, 'total_pnl': total_pnl, 'total_pnl_percent': total_pnl_percent, 'sharpe_ratio': random.uniform(0.5,2.0) if portfolio['assets'] else 0.0})


# --- Data Simulation Function ---
def get_mock_price_for_asset(api_id: str, source_hint: Optional[str] = None) -> float:
    api_id_upper = api_id.upper()
    # Forex
    if "EUR/USD" == api_id_upper: return random.uniform(1.05, 1.15)
    if "USD/JPY" == api_id_upper: return random.uniform(130, 150)
    if "GBP/USD" == api_id_upper: return random.uniform(1.20, 1.30)
    if "USD/CHF" == api_id_upper: return random.uniform(0.88, 0.95)
    if "AUD/USD" == api_id_upper: return random.uniform(0.65, 0.75)
    if "USD/CAD" == api_id_upper: return random.uniform(1.30, 1.40)
    if "NZD/USD" == api_id_upper: return random.uniform(0.58, 0.68)
    if "EUR/GBP" == api_id_upper: return random.uniform(0.84, 0.89)
    if "EUR/JPY" == api_id_upper: return random.uniform(140, 160)
    if "GBP/JPY" == api_id_upper: return random.uniform(160, 180)
    if "AUD/JPY" == api_id_upper: return random.uniform(85, 95)
    if "CHF/JPY" == api_id_upper: return random.uniform(140, 160)
    if "USD/TRY" == api_id_upper: return random.uniform(25, 35) # Volatile
    if "USD/ZAR" == api_id_upper: return random.uniform(17, 20)
    # Major Cryptos (using symbol or /USD convention)
    if api_id_upper in ["BTC", "BITCOIN", "BTC/USD"]: return random.uniform(40000, 70000)
    if api_id_upper in ["ETH", "ETHEREUM", "ETH/USD"]: return random.uniform(2000, 4000)
    if api_id_upper in ["BNB", "BNB/USD"]: return random.uniform(200, 400)
    if api_id_upper in ["SOL", "SOLANA", "SOL/USD"]: return random.uniform(50, 200)
    if api_id_upper in ["XRP", "XRP/USD"]: return random.uniform(0.3, 0.7)
    if api_id_upper in ["ADA", "CARDANO", "ADA/USD"]: return random.uniform(0.2, 0.5)
    if api_id_upper in ["AVAX", "AVAX/USD"]: return random.uniform(10, 50)
    if api_id_upper in ["DOT", "DOT/USD"]: return random.uniform(4, 10)
    if api_id_upper in ["DOGE", "DOGE/USD"]: return random.uniform(0.05, 0.15)
    if api_id_upper in ["TON", "TONCOIN", "TON/USD"]: return random.uniform(1, 3) # TON
    # AI & Big Data
    if api_id_upper == "RNDR": return random.uniform(1, 5)
    if api_id_upper == "FET": return random.uniform(0.2, 0.8)
    if api_id_upper == "OCEAN": return random.uniform(0.1, 0.5)
    if api_id_upper == "GRT": return random.uniform(0.05, 0.2)
    # Privacy
    if api_id_upper == "XMR": return random.uniform(100, 200)
    if api_id_upper == "ZEC": return random.uniform(20, 60)
    # Stablecoins
    if api_id_upper in ["USDT", "USDC", "DAI", "TUSD"]: return random.uniform(0.99, 1.01)
    # Gaming & Metaverse
    if api_id_upper == "AXS": return random.uniform(5, 15)
    if api_id_upper == "SAND": return random.uniform(0.3, 0.8)
    if api_id_upper == "MANA": return random.uniform(0.2, 0.7)
    if api_id_upper == "GALA": return random.uniform(0.01, 0.05)
    # DeFi
    if api_id_upper == "UNI": return random.uniform(3, 8)
    if api_id_upper == "AAVE": return random.uniform(50, 100)
    if api_id_upper == "CRV": return random.uniform(0.4, 1.0)
    if api_id_upper == "COMP": return random.uniform(30, 70)
    # Stocks
    if "AAPL" == api_id_upper: return random.uniform(150, 250)
    if "MSFT" == api_id_upper: return random.uniform(300, 450)
    # Generic fallback
    print(f"No specific mock price for {api_id}, using generic fallback.")
    return random.uniform(1, 500)

# --- FastAPI App Initialization ---
app = FastAPI(title="Ultimate Stock/Crypto Prediction API", version="1.3.0") # Note version bump
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
api_router_v1 = "/api/v1"

# --- Auth & User Endpoints ---
@app.post(f"{api_router_v1}/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict or not verify_password(form_data.password, user_dict["hashed_password"]) or user_dict.get(
            "disabled"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Incorrect email, password, or inactive user")
    access_token = create_access_token(data={"sub": user_dict["email"], "name": user_dict.get("full_name")})
    return {"access_token": access_token, "token_type": "bearer", "user_name": user_dict.get("full_name")}


@app.post(f"{api_router_v1}/auth/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(user_in: UserCreate):
    if user_in.email in fake_users_db: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                                           detail="Email already registered")
    portfolio_id = f"{user_in.email.replace('@', '_').replace('.', '_')}_portfolio"
    new_user_data = {"email": user_in.email, "full_name": user_in.full_name,
                     "hashed_password": get_password_hash(user_in.password), "disabled": False,
                     "portfolio_id": portfolio_id, "alerts": [], "preferences": UserPreferences().model_dump()}
    fake_users_db[user_in.email] = new_user_data
    fake_portfolios_db[portfolio_id] = {
        "summary": {"total_value": 0, "total_pnl": 0, "total_pnl_percent": 0, "sharpe_ratio": 0}, "assets": []}
    return User(**new_user_data)


@app.get(f"{api_router_v1}/users/me", response_model=User) # Existing endpoint
async def read_users_me(current_user_data: dict = Depends(get_current_firebase_user)):
    # current_user_data is from fake_users_db, which should have preferences
    return User(**current_user_data)


@app.put(f"{api_router_v1}/users/me/preferences", response_model=UserPreferences)
async def update_user_preferences(prefs: UserPreferences,
                                  current_user: UserInDBBase = Depends(get_current_active_user)):
    fake_users_db[current_user.email]["preferences"] = prefs.model_dump()
    return prefs


# --- API Client Service ---
TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
API_TIMEOUT = 10


async def fetch_coinmarketcap_latest_quote(symbol_or_id: str) -> Optional[Dict[str, Any]]:
    if not COINMARKETCAP_API_KEY:
        print("WARNING: COINMARKETCAP_API_KEY not set. Cannot fetch from CoinMarketCap.")
        return None

    # CoinMarketCap API v1/cryptocurrency/quotes/latest can take 'id' or 'symbol'
    # We'll assume `symbol_or_id` is the uppercase symbol like "BTC", "ETH"
    # Or a comma-separated list of symbols. For single price, just one.
    # For mapping 'bitcoin' (apiId) to 'BTC' (symbol), we might need a map or rely on the frontend passing the correct symbol.
    # Let's assume apiId from frontend for CoinMarketCap sourceHint will be the symbol (e.g., "BTC", "ETH").

    url = f"{COINMARKETCAP_BASE_URL}/v1/cryptocurrency/quotes/latest"
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY,
    }
    # Prefer using CoinMarketCap ID if known, otherwise symbol.
    # For simplicity, if apiId is numeric, assume it's ID, else symbol.
    params = {}
    if symbol_or_id.isdigit():
        params['id'] = symbol_or_id
    else:
        params['symbol'] = symbol_or_id.upper()  # CMC usually expects uppercase symbols
    params['convert'] = 'USD'  # Get price in USD

    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status", {}).get("error_code") == 0:
                # Data for the first (and likely only) symbol requested
                # The key in 'data' dict is the symbol if 'symbol' param was used, or ID if 'id' param was used.
                # We need to find the actual data entry.
                asset_data = None
                if 'data' in data and data['data']:
                    # If by symbol, key is symbol. If by ID, key is ID (as string).
                    first_key = list(data['data'].keys())[0]
                    asset_data = data['data'].get(first_key)

                if asset_data:
                    return asset_data  # Contains name, symbol, quote.USD.price etc.
                else:
                    print(f"CoinMarketCap: Data not found for {symbol_or_id} in response: {data}")
                    return None
            else:
                print(f"CoinMarketCap API error for {symbol_or_id}: {data.get('status')}")
                return None
        except httpx.HTTPStatusError as e:
            print(f"CoinMarketCap HTTP error for {symbol_or_id}: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            print(f"CoinMarketCap request error for {symbol_or_id}: {e}")
            return None
        except Exception as e:
            print(f"General error fetching CoinMarketCap latest quote for {symbol_or_id}: {e}")
            return None

async def fetch_coinmarketcap_historical_simulated(api_id_symbol: str, days: str = "1") -> Dict[str, Any]:
    # THIS IS SIMULATED as real CMC historical OHLCV is often paid.
    # We'll use the latest quote and generate a mock historical trend based on it.
    print(f"Simulating CoinMarketCap historical for {api_id_symbol} for {days} days")
    latest_quote_data = await fetch_coinmarketcap_latest_quote(api_id_symbol)
    base_price = get_mock_price_for_asset(api_id_symbol, 'coinmarketcap_mock')  # Fallback if quote fails

    if latest_quote_data and latest_quote_data.get('quote', {}).get('USD', {}).get('price'):
        base_price = latest_quote_data['quote']['USD']['price']

    num_points_map = {"1": 288, "7": 168, "30": 30 * 24, "max": 365 * 2}  # rough points for periods
    num_points = num_points_map.get(days, 288)

    end_time = int(time.time())
    # Calculate start_time based on 'days' (string like "1", "7", "30")
    try:
        days_int = int(days) if days != "max" else 365 * 2  # Max 2 years for mock
    except ValueError:
        days_int = 1
    start_time = end_time - (days_int * 24 * 60 * 60)

    timestamps = [int(start_time + i * (end_time - start_time) / num_points) for i in range(num_points)]
    prices = [base_price] * num_points
    for i in range(1, num_points):
        prices[i] = max(0.00001, prices[i - 1] * (1 + random.uniform(-0.025, 0.025)))  # Simulate daily volatility
    return {"asset_id": api_id_symbol, "timestamps": timestamps, "prices": prices,
            "source": "coinmarketcap_simulated_historical"}


async def fetch_twelvedata_historical_data(symbol: str, interval: str = "1day", outputsize: int = 30) -> Dict[str, Any]:
    if not TWELVEDATA_API_KEY: raise HTTPException(status_code=500, detail="Twelve Data API key not configured.")
    url = f"{TWELVEDATA_BASE_URL}/time_series"; params = { "symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": TWELVEDATA_API_KEY, "format": "JSON", "timezone": "UTC" }
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try: response = await client.get(url, params=params); response.raise_for_status(); data = response.json();
        except httpx.HTTPStatusError as e: print(f"TwelveData HTTP error for {symbol}: {e.response.status_code} - {e.response.text}"); raise HTTPException(status_code=e.response.status_code, detail=f"TwelveData API error: {e.response.json().get('message', 'Unknown error') if e.response.content else 'Unknown TwelveData API error'}")
        except httpx.RequestError as e: print(f"TwelveData request error for {symbol}: {e}"); raise HTTPException(status_code=503, detail=f"Could not connect to TwelveData API: {e}")
        except Exception as e: print(f"General error fetching TwelveData data for {symbol}: {e}"); raise HTTPException(status_code=500, detail="Error processing data from TwelveData.")
        if data.get("status") == "error": raise HTTPException(status_code=data.get("code", 400), detail=data.get("message", "Twelve Data API error"))
        values = data.get("values", []);
        if not values: return {"asset_id": symbol, "timestamps": [], "prices": [], "source": "twelvedata_empty"}
        values.reverse()
        timestamps = []; prices = []
        for item in values:
            try: dt_format = "%Y-%m-%d %H:%M:%S" if len(item["datetime"]) > 10 else "%Y-%m-%d"; timestamps.append(int(datetime.strptime(item["datetime"], dt_format).timestamp())); prices.append(float(item["close"]))
            except ValueError: print(f"Skipping value with unparseable datetime: {item['datetime']}"); continue
        return {"asset_id": symbol, "timestamps": timestamps, "prices": prices, "source": "twelvedata"}


async def get_live_price_from_api(api_id: str, source_hint: Optional[str] = None) -> float:
    if source_hint == 'coinmarketcap':
        quote_data = await fetch_coinmarketcap_latest_quote(api_id)  # api_id here is expected to be symbol like BTC
        if quote_data and quote_data.get('quote', {}).get('USD', {}).get('price'):
            return float(quote_data['quote']['USD']['price'])
        else:
            print(f"Could not get live price from CoinMarketCap for {api_id}. Falling back to mock.")
            return get_mock_price_for_asset(api_id, 'coinmarketcap_mock')
    elif source_hint == 'twelvedata' and TWELVEDATA_API_KEY:
        # ... (TwelveData live price logic as before) ...
        td_symbol = api_id
        if api_id.lower() == 'bitcoin':
            td_symbol = 'BTC/USD'
        elif api_id.lower() == 'ethereum':
            td_symbol = 'ETH/USD'
        elif api_id.lower() == 'solana':
            td_symbol = 'SOL/USD'
        url = f"{TWELVEDATA_BASE_URL}/price?symbol={td_symbol}&apikey={TWELVEDATA_API_KEY}"
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            try:
                response = await client.get(
                    url); response.raise_for_status(); data = response.json(); price_str = data.get(
                    "price"); return float(price_str) if price_str is not None else get_mock_price_for_asset(api_id,
                                                                                                             source_hint)
            except Exception as e:
                print(
                    f"Live price error TwelveData for {td_symbol} (orig: {api_id}): {e}"); return get_mock_price_for_asset(
                    api_id, source_hint)

    print(f"Using mock live price for {api_id} (source hint: {source_hint})")
    return get_mock_price_for_asset(api_id, source_hint)

# --- Modified Endpoints to use API Clients ---
@app.get(f"{api_router_v1}/data/historical/{'{api_id}'}")
async def get_historical_data_endpoint(api_id: str, period: str = "1d", source_hint: Optional[str] = None):
    print(f"API call: Historical data for {api_id}, period {period}, source hint {source_hint}")
    try:
        if source_hint == 'coinmarketcap':
            # Map 'period' to CoinMarketCap 'days' parameter if their actual historical API was used
            # For our simulated one, it takes 'days' as string like "1", "7", "30"
            days_map = {"1h": "1", "1d": "1", "7d": "7", "1mo": "30", "90d":"90", "1y": "365", "max": "max"}
            cmc_days_param = days_map.get(period, "1")
            return await fetch_coinmarketcap_historical_simulated(api_id, days=cmc_days_param)
        elif source_hint == 'twelvedata':
            # ... (TwelveData logic as before) ...
            interval_map = {"1h": "1min", "1d": "1day", "7d": "1day", "1mo": "1day"}
            outputsize_map = {"1h": 60, "1d": 30, "7d": 7, "1mo": 30}
            td_interval = interval_map.get(period, "1day")
            td_outputsize = outputsize_map.get(period, 30)
            td_symbol = api_id
            if api_id.lower() == 'bitcoin': td_symbol = 'BTC/USD'
            elif api_id.lower() == 'ethereum': td_symbol = 'ETH/USD'
            elif api_id.lower() == 'solana': td_symbol = 'SOL/USD'
            return await fetch_twelvedata_historical_data(td_symbol, interval=td_interval, outputsize=td_outputsize)
        else: # Fallback to mock
            # ... (Mock logic as before) ...
            print(f"Warning: Using mock historical data for {api_id} (source_hint: {source_hint}).")
            end_time = int(time.time()); points_map_fb = {"1h": 60, "1d": 288, "1w": 168, "1mo": 720}; seconds_map_fb = {"1h": 3600, "1d": 24*3600, "1w": 7*24*3600, "1mo": 30*24*3600}
            points_fb = points_map_fb.get(period, 288); start_time_fb = end_time - seconds_map_fb.get(period, 24*3600)
            timestamps_fb = [int(start_time_fb + i * (end_time - start_time_fb) / points_fb) for i in range(points_fb)]
            base_price_fb = get_mock_price_for_asset(api_id, source_hint)
            prices_fb = [base_price_fb] * points_fb
            for i in range(1, points_fb): prices_fb[i] = max(0.00001 if "/" in api_id else 0.01, prices_fb[i-1] + random.uniform(-prices_fb[i-1]*0.015, prices_fb[i-1]*0.015) + (base_price_fb * 0.001) * (i/points_fb - 0.5))
            return {"asset_id": api_id, "timestamps": timestamps_fb, "prices": prices_fb, "source": "mock_fallback"}
    except HTTPException as e: raise e
    except Exception as e: print(f"Error in get_historical_data_endpoint for {api_id}: {e}"); raise HTTPException(status_code=500, detail=f"Failed to fetch historical data for {api_id}.")

# ... (The rest of your FastAPI app: WebSocket, other endpoints for ML, Portfolio, Alerts etc.)
# Ensure all definitions and indentations are correct for the remaining part of the file.
# I'll include the definition for _recalculate_portfolio_summary as it was specifically requested.

# WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, asset_key: str):
        await websocket.accept();
        self.active_connections.setdefault(asset_key, []).append(websocket)

    def disconnect(self, websocket: WebSocket, asset_key: str):
        if asset_key in self.active_connections and websocket in self.active_connections[asset_key]:
            self.active_connections[asset_key].remove(websocket);
        if asset_key in self.active_connections and not self.active_connections[asset_key]:
            del self.active_connections[asset_key]

    async def broadcast(self, message: str, asset_key: str):
        if asset_key in self.active_connections:
            for connection in list(self.active_connections[asset_key]):
                try:
                    await connection.send_text(message)
                except RuntimeError:
                    self.disconnect(connection, asset_key)


manager = ConnectionManager()
live_data_task = None


async def simulate_live_data_broadcast():
    while True:
        await asyncio.sleep(random.uniform(5, 10))
        active_asset_keys = list(manager.active_connections.keys())
        if not active_asset_keys: continue
        print(f"Polling live prices for: {active_asset_keys}")
        for asset_key in active_asset_keys:
            api_id, _, source_hint_val = asset_key.partition("?sourceHint=")
            # Improved logic to handle 'None' as a string from query params
            source_hint = source_hint_val if source_hint_val and source_hint_val != 'None' else None
            try:
                new_price = await get_live_price_from_api(api_id, source_hint)
                data_dict = {"asset_id": api_id, "price": new_price, "timestamp": int(time.time())};
                data_str = str(data_dict).replace("'", '"')
                await manager.broadcast(data_str, asset_key)
            except Exception as e:
                print(f"Error polling live price for {asset_key}: {e}")


@app.on_event("startup")
async def startup_event(): global live_data_task; live_data_task = asyncio.create_task(simulate_live_data_broadcast())


@app.on_event("shutdown")
async def shutdown_event():
    if live_data_task: live_data_task.cancel();
    try:
        await live_data_task
    except asyncio.CancelledError:
        print("Live data task cancelled.")
    except Exception as e:
        print(f"Error during live_data_task shutdown: {e}")


@app.websocket("/ws/livedata/{api_id}")
async def websocket_endpoint(websocket: WebSocket, api_id: str, source_hint: Optional[str] = None):
    asset_key = f"{api_id}?sourceHint={source_hint}" if source_hint else api_id
    await manager.connect(websocket, asset_key)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        print(f"WebSocket client disconnected for {asset_key}")
    except Exception as e:
        print(f"WebSocket error for {asset_key}: {e}")
    finally:
        manager.disconnect(websocket, asset_key)


# --- Other Endpoints ---

# Market News
@app.get(f"{api_router_v1}/market/news", response_model=MarketNewsResponse)
async def get_market_news(limit: int = 10):
    mock_news = [];
    sources = ["CryptoNewsPro", "ForexInsider", "MarketWatchSim", "TechInvestToday"];
    keywords = ["Bitcoin", "Ethereum", "USD", "Tech Stocks", "Global Economy", "Forex Trends"]
    for i in range(limit):
        source = random.choice(sources);
        keyword = random.choice(keywords)
        mock_news.append(MarketNewsArticle(
            title=f"Market Update: {keyword} Analysis & Outlook - #{random.randint(100, 999)}",
            description=f"Latest insights on {keyword}. Market sentiment appears {random.choice(['cautiously optimistic', 'bearish short-term', 'volatile but with long-term potential'])}. Key support and resistance levels identified.",
            url=f"https://mock-news-provider.com/article/{keyword.lower().replace(' ', '-')}-{random.randint(1000, 9999)}",
            source=source,
            published_at=datetime.utcnow() - timedelta(minutes=random.randint(5, 360))
        ))
    return MarketNewsResponse(articles=mock_news)


# ML Prediction
@app.post(f"{api_router_v1}/predict/{'{api_id}'}", response_model=PredictionResponse)
async def get_prediction(api_id: str, request: PredictionRequest, source_hint: Optional[str] = None,
                         current_user: UserInDBBase = Depends(get_current_active_user)):
    await asyncio.sleep(random.uniform(0.5, 1.5));
    trend = random.choice(["UP", "DOWN", "SIDEWAYS"])
    try:
        current_price_data = await get_historical_data_endpoint(api_id, period="1h", source_hint=source_hint)
        current_price = current_price_data["prices"][-1] if current_price_data[
            "prices"] else await get_live_price_from_api(api_id, source_hint)
    except Exception as e:
        print(f"Error getting price for prediction of {api_id}: {e}")
        current_price = await get_live_price_from_api(api_id, source_hint)
    change_factor = random.uniform(-0.03, 0.03) if request.time_horizon == "short-term" else random.uniform(-0.1, 0.1);
    predicted_value = max(0.00001 if "/" in api_id else 0.01, current_price * (1 + change_factor));
    conf_factor = 0.02 if request.time_horizon == "short-term" else 0.05
    return PredictionResponse(asset_id=api_id, prediction={"trend": trend, "value": predicted_value,
                                                           "timestamp": int(time.time()) + (
                                                               3600 if request.time_horizon == "short-term" else 24 * 3600)},
                              confidence_interval={"lower": predicted_value * (1 - conf_factor),
                                                   "upper": predicted_value * (1 + conf_factor)})


# Analysis Tools
@app.get(f"{api_router_v1}/analysis/technical/{'{api_id}'}", response_model=Dict[str, Any])
async def get_technical_analysis(api_id: str, source_hint: Optional[str] = None):
    await asyncio.sleep(0.5);
    hist_data = await get_historical_data_endpoint(api_id, "1h", source_hint=source_hint)
    prices = hist_data["prices"];
    num_points = len(prices)
    sma50_mock = [
        random.uniform(prices[i] * 0.95, prices[i] * 1.05) if num_points > i and prices[i] is not None else None for i
        in range(num_points)]
    ema20_mock = [
        random.uniform(prices[i] * 0.97, prices[i] * 1.03) if num_points > i and prices[i] is not None else None for i
        in range(num_points)]
    current_price_for_bb = prices[-1] if prices and prices[-1] is not None else get_mock_price_for_asset(api_id,
                                                                                                         source_hint)
    bb_middle = current_price_for_bb * random.uniform(0.98, 1.02);
    bb_std_dev_mock = current_price_for_bb * random.uniform(0.02, 0.05)
    bollinger_bands_data = {"upper": bb_middle + 2 * bb_std_dev_mock, "middle": bb_middle,
                            "lower": bb_middle - 2 * bb_std_dev_mock}
    return {"asset_id": api_id, "indicators": TechnicalIndicators(
        MACD={"macd": random.uniform(-10, 10), "signal": random.uniform(-10, 10), "histogram": random.uniform(-2, 2)},
        RSI=random.uniform(20, 80), BollingerBands=bollinger_bands_data,
        MovingAverages={"sma50": sma50_mock, "ema20": ema20_mock}).model_dump(exclude_none=True)}


# Portfolio Management
#@app.get(f"{api_router_v1}/portfolio", response_model=PortfolioResponse)
#async def get_portfolio(current_user: UserInDBBase = Depends(get_current_active_user)):
@app.get(f"{api_router_v1}/portfolio", response_model=PortfolioResponse)
async def get_portfolio_endpoint(current_user: UserInDBBase = Depends(get_current_active_user)):
    portfolio_id = current_user.portfolio_id
    if portfolio_id not in fake_portfolios_db:
        fake_portfolios_db[portfolio_id] = {
            "summary": PortfolioSummary(total_value=0, total_pnl=0, total_pnl_percent=0, sharpe_ratio=0).model_dump(),
            "assets": []}

    for asset_dict in fake_portfolios_db[portfolio_id]["assets"]:
        api_id_for_fetch = asset_dict[
            "asset_id"]  # This should be the ID for the API (e.g. 'bitcoin', 'AAPL', 'EUR/USD')
        source_hint_portfolio = asset_dict.get("sourceHint")  # If you store sourceHint per asset
        if not source_hint_portfolio:  # Fallback source hint logic
            source_hint_portfolio = 'twelvedata' if asset_dict["type"] in ['stock', 'forex',
                                                                           'crypto'] else 'mock'  # Default to twelvedata for crypto too now

        try:
            live_price = await get_live_price_from_api(api_id_for_fetch, source_hint_portfolio)
            asset_dict["current_price"] = live_price
        except Exception as e:
            print(f"Error updating price for portfolio asset {api_id_for_fetch}: {e}")
            asset_dict["current_price"] = asset_dict.get("current_price", asset_dict["average_purchase_price"])

        asset_dict["current_value"] = asset_dict["current_price"] * asset_dict["quantity"]
        asset_dict["pnl"] = (asset_dict["current_price"] - asset_dict["average_purchase_price"]) * asset_dict[
            "quantity"]
        asset_dict["pnl_percent"] = ((asset_dict["current_price"] / asset_dict["average_purchase_price"]) - 1) if \
        asset_dict["average_purchase_price"] > 0.000001 else 0

    _recalculate_portfolio_summary(portfolio_id)  # This function is now defined
    return PortfolioResponse(**fake_portfolios_db[portfolio_id])




@app.post(f"{api_router_v1}/portfolio/assets", response_model=PortfolioAssetInDB, status_code=status.HTTP_201_CREATED)
async def add_asset_to_portfolio(asset_item: AssetPortfolioCreate,
                                 current_user: UserInDBBase = Depends(get_current_active_user)):
    portfolio_id = current_user.portfolio_id
    asset_id_in_db = f"{asset_item.asset_id.lower().replace('/', '_')}_{random.randint(1000, 9999)}"

    source_hint = 'coingecko' if asset_item.type == 'crypto' else 'twelvedata' if asset_item.type in ['stock',
                                                                                                      'forex'] else None

    try:
        current_price = await get_live_price_from_api(asset_item.asset_id, source_hint)
    except Exception:
        current_price = asset_item.purchase_price

    new_asset_entry = PortfolioAssetInDB(
        id=asset_id_in_db, asset_id=asset_item.asset_id, type=asset_item.type, quantity=asset_item.quantity,
        average_purchase_price=asset_item.purchase_price, current_price=current_price,
        current_value=current_price * asset_item.quantity,
        pnl=(current_price - asset_item.purchase_price) * asset_item.quantity,
        pnl_percent=((current_price / asset_item.purchase_price) - 1) if asset_item.purchase_price > 0.000001 else 0
    )

    fake_portfolios_db.setdefault(portfolio_id, {"assets": [], "summary": {}}).get("assets").append(
        new_asset_entry.model_dump())
    _recalculate_portfolio_summary(portfolio_id)
    return new_asset_entry


@app.delete(f"{api_router_v1}/portfolio/assets/{'{item_id}'}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_asset_from_portfolio(item_id: str, current_user: UserInDBBase = Depends(get_current_active_user)):
    portfolio_id = current_user.portfolio_id
    if portfolio_id not in fake_portfolios_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")

    assets_list = fake_portfolios_db[portfolio_id].get("assets", [])
    initial_len = len(assets_list)

    assets_list[:] = [asset for asset in assets_list if asset.get('id') != item_id]

    if len(assets_list) == initial_len:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset item not found in portfolio")

    _recalculate_portfolio_summary(portfolio_id)


# Alert Management
@app.get(f"{api_router_v1}/alerts", response_model=List[AlertResponse])
async def get_user_alerts(current_user: UserInDBBase = Depends(get_current_active_user)):
    return [AlertResponse(**alert_data) for alert_data in current_user.alerts]


@app.post(f"{api_router_v1}/alerts", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(alert_in: AlertCreate, current_user: UserInDBBase = Depends(get_current_active_user)):
    alert_id = f"alert_{random.randint(10000, 99999)}"
    new_alert = AlertResponse(id=alert_id, user_id=current_user.email, **alert_in.model_dump())

    fake_users_db[current_user.email]["alerts"].append(new_alert.model_dump())
    return new_alert


@app.delete(f"{api_router_v1}/alerts/{'{alert_id}'}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert(alert_id: str, current_user: UserInDBBase = Depends(get_current_active_user)):
    user_alerts = fake_users_db[current_user.email].get("alerts", [])
    initial_len = len(user_alerts)

    fake_users_db[current_user.email]["alerts"][:] = [alert for alert in user_alerts if alert.get('id') != alert_id]

    if len(fake_users_db[current_user.email]["alerts"]) == initial_len:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")


if __name__ == "__main__":
    import uvicorn

    print("Starting Uvicorn server for PredictoPro API...")
    print(
        f"TWELVEDATA_API_KEY is set: {'Yes' if TWELVEDATA_API_KEY else 'NO - Stock/Forex data will be mocked or fail.'}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
