# ============ IMPORTS ============
import os
import time
import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from decimal import Decimal
import numpy as np

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from web3 import Web3
from web3.middleware import geth_poa_middleware
from dotenv import load_dotenv
import ccxt

# ============ CONFIGURATIE KLASSEN ============
@dataclass
class Token:
    """Token configuratie"""
    symbol: str
    address: str
    decimals: int
    min_amount: float = 0.001
    
@dataclass
class Dex:
    """DEX configuratie"""
    name: str
    router_address: str
    factory_address: str
    fee: int  # in basis points (e.g., 30 = 0.3%)
    is_active: bool = True

@dataclass
class ArbitrageOpportunity:
    """Gedetecteerde arbitrage kans"""
    timestamp: datetime
    token_in: str
    token_out: str
    buy_dex: str
    sell_dex: str
    amount_in: float
    amount_out: float
    expected_profit: float
    profit_percentage: float
    gas_cost_eth: float
    net_profit: float
    confidence_score: float = 0.0
    status: str = "detected"  # detected, executing, completed, failed
    
# ============ INITIALISATIE ============
load_dotenv()

class ArbitrageBot:
    """Hoofdklasse voor de arbitrage bot"""
    
    def __init__(self):
        self.setup_page()
        self.load_config()
        self.init_web3()
        self.init_state()
        
    def setup_page(self):
        """Streamlit pagina configuratie"""
        st.set_page_config(
            page_title="🤖 Advanced Arbitrage Bot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header { font-size: 2.5rem; color: #1E88E5; }
        .profit-positive { color: #4CAF50; font-weight: bold; }
        .profit-negative { color: #F44336; font-weight: bold; }
        .metric-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def load_config(self):
        """Laad alle configuraties"""
        # Netwerk config
        self.network = st.secrets.get("NETWORK", "sepolia")
        self.test_mode = st.secrets.get("TEST_MODE", "True") == "True"
        
        # RPC URLs
        self.rpc_urls = {
            "mainnet": st.secrets.get("MAINNET_RPC_URL", ""),
            "sepolia": st.secrets.get("SEPOLIA_RPC_URL", "https://eth-sepolia.g.alchemy.com/v2/demo"),
            "arbitrum": st.secrets.get("ARBITRUM_RPC_URL", ""),
            "optimism": st.secrets.get("OPTIMISM_RPC_URL", "")
        }
        
        # Tokens (uitgebreide lijst)
        self.tokens = {
            "ETH": Token("ETH", "0x0000000000000000000000000000000000000000", 18, 0.01),
            "WETH": Token("WETH", "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18, 0.01),
            "USDC": Token("USDC", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606EB48", 6, 10),
            "USDT": Token("USDT", "0xdAC17F958D2ee523a2206206994597C13D831ec7", 6, 10),
            "DAI": Token("DAI", "0x6B175474E89094C44Da98b954EedeAC495271d0F", 18, 10),
            "WBTC": Token("WBTC", "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", 8, 0.001),
            "LINK": Token("LINK", "0x514910771AF9Ca656af840dff83E8264EcF986CA", 18, 1),
            "UNI": Token("UNI", "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", 18, 1),
            "AAVE": Token("AAVE", "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9", 18, 0.1)
        }
        
        # DEX configuraties
        self.dexes = {
            "uniswap_v2": Dex("Uniswap V2", 
                            "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                            "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                            30),
            "sushiswap": Dex("Sushiswap",
                           "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
                           "0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac",
                           30),
            "pancakeswap": Dex("PancakeSwap",
                             "0x10ED43C718714eb63d5aA57B78B54704E256024E",
                             "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
                             25),
            "curve": Dex("Curve Finance",
                        "0x7D86446dDb609eD0F5f8684AcF30380a356b2B4c",
                        "0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5",
                        4)  # 0.04%
        }
        
        # Actieve token paren voor monitoring
        self.monitored_pairs = [
            ("WETH", "USDC"),
            ("WETH", "USDT"),
            ("WETH", "DAI"),
            ("USDC", "USDT"),
            ("WBTC", "WETH"),
            ("LINK", "WETH"),
            ("UNI", "WETH")
        ]
        
        # Trading parameters
        self.min_profit_threshold = 0.5  # Minimale winst percentage
        self.max_slippage = 0.5  # Max slippage percentage
        self.max_gas_price_gwei = 100  # Maximale gas price
        self.min_trade_amount_usd = 100  # Minimale trade grootte
        self.max_concurrent_trades = 3  # Maximaal aantal parallelle trades
        
    def init_web3(self):
        """Initialiseer Web3 connecties"""
        try:
            rpc_url = self.rpc_urls.get(self.network)
            if not rpc_url:
                st.error(f"Geen RPC URL gevonden voor netwerk: {self.network}")
                return None
                
            self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 60}))
            
            # Voor Polygon/Arbitrum chains
            if self.network in ["arbitrum", "optimism", "polygon"]:
                self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if self.w3.is_connected():
                st.sidebar.success(f"✅ Verbonden met {self.network.upper()}")
                st.sidebar.info(f"Block: {self.w3.eth.block_number}")
                
                # Laad contract ABIs
                self.load_abis()
                
                # Initialiseer DEX contracts
                self.init_dex_contracts()
                
                return self.w3
            else:
                st.error("❌ Kan niet verbinden met blockchain")
                return None
                
        except Exception as e:
            st.error(f"Fout bij initialiseren Web3: {str(e)}")
            return None
    
    def load_abis(self):
        """Laad contract ABIs"""
        # Uniswap V2 Router ABI (vereenvoudigd)
        self.router_abi = [
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
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountOut", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"}
                ],
                "name": "getAmountsIn",
                "outputs": [
                    {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactETHForTokens",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "payable",
                "type": "function"
            }
        ]
    
    def init_dex_contracts(self):
        """Initialiseer DEX contracten"""
        self.dex_contracts = {}
        for dex_name, dex in self.dexes.items():
            if dex.is_active:
                try:
                    contract = self.w3.eth.contract(
                        address=self.w3.to_checksum_address(dex.router_address),
                        abi=self.router_abi
                    )
                    self.dex_contracts[dex_name] = contract
                except Exception as e:
                    st.warning(f"Kan {dex.name} contract niet laden: {str(e)}")
    
    def init_state(self):
        """Initialiseer Streamlit session state"""
        if 'opportunities' not in st.session_state:
            st.session_state.opportunities = []
        if 'trades' not in st.session_state:
            st.session_state.trades = []
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
        if 'auto_trading' not in st.session_state:
            st.session_state.auto_trading = False
        if 'monitoring_thread' not in st.session_state:
            st.session_state.monitoring_thread = None
        if 'price_data' not in st.session_state:
            st.session_state.price_data = {}
    
    # ============ MONITORING FUNCTIES ============
    
    async def monitor_prices(self):
        """Monitor prijzen voor alle token paren op alle DEXen"""
        while st.session_state.monitoring_active:
            try:
                opportunities = []
                
                for token_a, token_b in self.monitored_pairs:
                    # Skip als tokens hetzelfde zijn
                    if token_a == token_b:
                        continue
                    
                    # Zoek arbitrage tussen alle DEX combinaties
                    dex_combinations = self.get_dex_combinations()
                    
                    for buy_dex, sell_dex in dex_combinations:
                        opportunity
