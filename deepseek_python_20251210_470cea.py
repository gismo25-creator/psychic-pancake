# Ethereum Arbitrage Bot met UI (Streamlit)

import os
import time
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from web3 import Web3
from dotenv import load_dotenv
from datetime import datetime

# Load environment
load_dotenv()

# Connect to Ethereum node
QUICKNODE_HTTP = "https://orbital-frosty-bush.ethereum-hoodi.quiknode.pro/0490360fbe097aa617ce87bcf31ed4b051c1c36c/"
w3 = Web3(Web3.HTTPProvider(QUICKNODE_HTTP))

# Routers
UNISWAP_ROUTER = Web3.to_checksum_address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")
SUSHISWAP_ROUTER = Web3.to_checksum_address("0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F")

# Tokens
TOKENS = {
    "WETH": Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"),
    "DAI": Web3.to_checksum_address("0x6B175474E89094C44Da98b954EedeAC495271d0F"),
    "USDC": Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606EB48"),
    "USDT": Web3.to_checksum_address("0xdAC17F958D2ee523a2206206994597C13D831ec7"),
    "WBTC": Web3.to_checksum_address("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"),
    "ETH": "ETH"
}

# ABI for Uniswap/Sushiswap
# Laad ABI
@st.cache_resource
def load_abi():
    """Laad contract ABI"""
    try:
        with open("router_abi.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback ABI voor Uniswap Router
        return [
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [
                    {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]

ROUTER_ABI = load_abi()

# Initialiseer contracts
uni_router = w3.eth.contract(address=UNISWAP_ROUTER, abi=ROUTER_ABI)
sushi_router = w3.eth.contract(address=SUSHISWAP_ROUTER, abi=ROUTER_ABI)


# Load router contracts
uni_router = w3.eth.contract(address=UNISWAP_ROUTER, abi=ROUTER_ABI)
sushi_router = w3.eth.contract(address=SUSHISWAP_ROUTER, abi=ROUTER_ABI)

# Account details (assume MetaMask account)
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
ACCOUNT_ADDRESS = w3.to_checksum_address(os.getenv("ACCOUNT_ADDRESS"))

# Voorbeeld secrets.toml (plaats dit in .streamlit/secrets.toml)
# PRIVATE_KEY = "0xabc123..."
# ACCOUNT_ADDRESS = "0xYourWalletAddressHere"

# Arbitrage function
def check_arbitrage(tokenA, tokenB, amount):
    path = [TOKENS[tokenA], TOKENS[tokenB]]

    try:
        uni_amounts = uni_router.functions.getAmountsOut(amount, path).call()
        sushi_amounts = sushi_router.functions.getAmountsOut(amount, path).call()

        uni_out = uni_amounts[-1] / (10 ** 18)
        sushi_out = sushi_amounts[-1] / (10 ** 18)

        diff = sushi_out - uni_out
        profit_percent = (diff / uni_out) * 100 if uni_out != 0 else 0

        return {
            "tokenA": tokenA,
            "tokenB": tokenB,
            "uni_out": uni_out,
            "sushi_out": sushi_out,
            "profit": diff,
            "profit_percent": profit_percent,
            "timestamp": datetime.now()
        }
    except:
        return None

# Execute swap (simplified example using Uniswap)
def execute_trade(trade_amount_eth):
    try:
        amount_in_wei = Web3.to_wei(trade_amount_eth, 'ether')
        token_in = TOKENS["WETH"]
        token_out = TOKENS["DAI"]
        deadline = int(time.time()) + 60 * 10

        tx = uni_router.functions.swapExactETHForTokens(
            0,
            [token_in, token_out],
            ACCOUNT_ADDRESS,
            deadline
        ).build_transaction({
            'from': ACCOUNT_ADDRESS,
            'value': amount_in_wei,
            'gas': 250000,
            'gasPrice': w3.to_wei('30', 'gwei'),
            'nonce': w3.eth.get_transaction_count(ACCOUNT_ADDRESS),
        })

        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        st.success(f"✅ Trade uitgevoerd: https://etherscan.io/tx/{tx_hash.hex()}")
    except Exception as e:
        st.error(f"❌ Fout bij uitvoeren trade: {str(e)}")

# Streamlit UI
st.set_page_config(page_title="Ethereum Arbitrage Bot", layout="wide")
st.title("🤖 Ethereum Arbitrage Bot")

run_bot = st.toggle("🔁 Start automatisch monitoren en handelen")

trade_amount_eth = st.slider("💰 Trading bedrag (ETH)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)

col1, col2 = st.columns(2)
token_pairs = [("WETH", "DAI"), ("USDC", "USDT"), ("WBTC", "ETH")]

log_data = []

if run_bot:
    with st.spinner("Zoeken naar arbitrage kansen..."):
        for i in range(100):  # Max 100 iteraties voor demo
            for pair in token_pairs:
                result = check_arbitrage(pair[0], pair[1], Web3.to_wei(trade_amount_eth, 'ether'))
                if result and result['profit'] > 0.01:  # Drempel voor winst
                    log_data.append(result)
                    st.success(f"💸 Arbitrage kans: {result['tokenA']} → {result['tokenB']}: +{result['profit_percent']:.2f}%")
                    execute_trade(trade_amount_eth)
                else:
                    st.info(f"{pair[0]} → {pair[1]}: Geen winst")
            time.sleep(10)

if log_data:
    df = pd.DataFrame(log_data)
    st.subheader("📈 Arbitragegeschiedenis")
    st.dataframe(df)

    fig = px.line(df, x="timestamp", y="profit_percent", color="tokenA",
                  title="Arbitrage Winstpercentage Over Tijd")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Nog geen arbitrage kansen gevonden...")
