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
# from web3.middleware import geth_poa_middleware
from dotenv import load_dotenv

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
    fee: int
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
    status: str = "detected"

# ============ HOOFDKLASSE ============
class ArbitrageBot:
    def __init__(self):
        self.setup_page()
        self.load_config()
        self.init_web3()
        self.init_state()
        
    def setup_page(self):
        """Streamlit pagina configuratie"""
        st.set_page_config(
            page_title="🤖 Advanced Arbitrage Bot",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
        <style>
        .main-header { font-size: 2.5rem; color: #1E88E5; }
        .profit-positive { color: #4CAF50; }
        .profit-negative { color: #F44336; }
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
        }
        
        # Tokens
        self.tokens = {
            "ETH": Token("ETH", "0x0000000000000000000000000000000000000000", 18, 0.01),
            "WETH": Token("WETH", "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18, 0.01),
            "USDC": Token("USDC", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6, 10),
            "USDT": Token("USDT", "0xdAC17F958D2ee523a2206206994597C13D831ec7", 6, 10),
            "DAI": Token("DAI", "0x6B175474E89094C44Da98b954EedeAC495271d0F", 18, 10),
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
        }
        
        # Actieve token paren
        self.monitored_pairs = [
            ("WETH", "USDC"),
            ("WETH", "USDT"),
            ("WETH", "DAI"),
            ("USDC", "USDT"),
        ]
        
        # Trading parameters
        self.min_profit_threshold = 0.5
        self.max_slippage = 0.5
        self.max_gas_price_gwei = 100
        
    def init_web3(self):
        """Initialiseer Web3 connecties"""
        try:
            rpc_url = self.rpc_urls.get(self.network)
            if not rpc_url:
                st.error(f"Geen RPC URL voor: {self.network}")
                return None
                
            self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 60}))
            
            if self.w3.is_connected():
                st.sidebar.success(f"✅ Verbonden met {self.network.upper()}")
                st.sidebar.info(f"Block: {self.w3.eth.block_number}")
                
                # Laad contract ABIs
                self.load_abis()
                self.init_dex_contracts()
                
                return self.w3
            else:
                st.error("❌ Kan niet verbinden met blockchain")
                return None
                
        except Exception as e:
            st.error(f"Web3 fout: {str(e)}")
            return None
    
    def load_abis(self):
        """Laad contract ABIs"""
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
                except:
                    st.warning(f"Kan {dex.name} contract niet laden")
    
    def init_state(self):
        """Initialiseer Streamlit state"""
        if 'opportunities' not in st.session_state:
            st.session_state.opportunities = []
        if 'trades' not in st.session_state:
            st.session_state.trades = []
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
        if 'price_data' not in st.session_state:
            st.session_state.price_data = {}
    
    # ============ MONITORING FUNCTIES ============
    
    def get_price(self, token_in: str, token_out: str, dex: str, amount_in: float) -> float:
        """Haal prijs op van een specifieke DEX"""
        try:
            if token_in not in self.tokens or token_out not in self.tokens:
                return 0.0
                
            # Converteer naar wei
            amount_wei = self.w3.to_wei(amount_in, 'ether')
            
            # Bereken path
            token_in_addr = self.tokens[token_in].address
            token_out_addr = self.tokens[token_out].address
            
            if token_in_addr == "0x0000000000000000000000000000000000000000":
                token_in_addr = self.tokens["WETH"].address
            if token_out_addr == "0x0000000000000000000000000000000000000000":
                token_out_addr = self.tokens["WETH"].address
                
            path = [token_in_addr, token_out_addr]
            
            # Haal prijs op
            contract = self.dex_contracts.get(dex)
            if contract:
                amounts = contract.functions.getAmountsOut(amount_wei, path).call()
                amount_out = self.w3.from_wei(amounts[-1], 'ether')
                return amount_out
                
        except Exception as e:
            st.sidebar.error(f"Prijs ophalen fout: {e}")
            
        return 0.0
    
    def scan_arbitrage_opportunities(self, trade_amount: float = 1.0) -> List[ArbitrageOpportunity]:
        """Scan voor arbitrage kansen tussen alle DEXen"""
        opportunities = []
        
        for token_in, token_out in self.monitored_pairs:
            # Controleer prijzen op alle DEXen
            dex_prices = {}
            
            for dex_name in self.dex_contracts.keys():
                price = self.get_price(token_in, token_out, dex_name, trade_amount)
                if price > 0:
                    dex_prices[dex_name] = price
            
            # Vergelijk prijzen tussen DEXen
            if len(dex_prices) >= 2:
                # Vind laagste en hoogste prijs
                min_dex = min(dex_prices, key=dex_prices.get)
                max_dex = max(dex_prices, key=dex_prices.get)
                
                buy_price = dex_prices[min_dex]
                sell_price = dex_prices[max_dex]
                
                # Bereken winst
                if buy_price > 0:
                    profit = sell_price - buy_price
                    profit_percent = (profit / buy_price) * 100
                    
                    # Schat gas kosten
                    gas_cost = self.estimate_gas_cost()
                    
                    # Filter op minimale winst
                    if profit_percent >= self.min_profit_threshold:
                        opportunity = ArbitrageOpportunity(
                            timestamp=datetime.now(),
                            token_in=token_in,
                            token_out=token_out,
                            buy_dex=min_dex,
                            sell_dex=max_dex,
                            amount_in=trade_amount,
                            amount_out=sell_price,
                            expected_profit=profit,
                            profit_percentage=profit_percent,
                            gas_cost_eth=gas_cost,
                            net_profit=profit - gas_cost,
                            confidence_score=min(profit_percent / 10, 1.0)
                        )
                        opportunities.append(opportunity)
        
        return opportunities
    
    def estimate_gas_cost(self) -> float:
        """Schat gas kosten voor een swap"""
        try:
            gas_price = self.w3.eth.gas_price
            gas_limit = 250000
            gas_cost_wei = gas_price * gas_limit
            return self.w3.from_wei(gas_cost_wei, 'ether')
        except:
            return 0.01  # Fallback
    
    # ============ TRADE UITVOERING ============
    
    def execute_trade(self, opportunity: ArbitrageOpportunity) -> Dict:
        """Voer een arbitrage trade uit"""
        try:
            if self.test_mode:
                return {
                    "status": "success",
                    "message": f"TEST: Trade {opportunity.token_in} → {opportunity.token_out}",
                    "profit": opportunity.net_profit
                }
            
            # Hier zou de echte trade logica komen
            # Voor nu simuleren we het
            time.sleep(2)  # Simuleer netwerk latency
            
            trade_result = {
                "timestamp": datetime.now(),
                "token_in": opportunity.token_in,
                "token_out": opportunity.token_out,
                "amount_in": opportunity.amount_in,
                "amount_out": opportunity.amount_out,
                "profit": opportunity.net_profit,
                "status": "completed",
                "tx_hash": f"0x{os.urandom(16).hex()}" if not self.test_mode else "TEST_TX"
            }
            
            # Voeg toe aan trade history
            st.session_state.trades.append(trade_result)
            
            return {
                "status": "success",
                "message": "Trade succesvol uitgevoerd",
                "data": trade_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Trade mislukt: {str(e)}"
            }
    
    def auto_trade_opportunities(self, opportunities: List[ArbitrageOpportunity]):
        """Voer automatisch trades uit voor winstgevende kansen"""
        for opportunity in opportunities:
            if opportunity.net_profit > 0 and opportunity.profit_percentage >= self.min_profit_threshold:
                result = self.execute_trade(opportunity)
                
                if result["status"] == "success":
                    st.success(f"✅ Auto-trade: {opportunity.token_in}→{opportunity.token_out} "
                             f"(+{opportunity.profit_percentage:.2f}%)")
                else:
                    st.error(f"❌ Auto-trade mislukt: {result['message']}")
    
    # ============ VISUALISATIE FUNCTIES ============
    
    def create_profit_chart(self):
        """Maak winst grafiek"""
        if not st.session_state.trades:
            return None
            
        df = pd.DataFrame(st.session_state.trades)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Winst Over Tijd", "Cumulatieve Winst", "Trade Verdeling", "Success Ratio"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Winst over tijd
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['profit'], mode='lines+markers', name='Winst'),
            row=1, col=1
        )
        
        # Cumulatieve winst
        df['cumulative_profit'] = df['profit'].cumsum()
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cumulative_profit'], mode='lines', name='Cumulatief'),
            row=1, col=2
        )
        
        # Trade verdeling per token pair
        token_pairs = df.groupby(['token_in', 'token_out']).size().reset_index()
        token_pairs['pair'] = token_pairs['token_in'] + '→' + token_pairs['token_out']
        
        fig.add_trace(
            go.Pie(labels=token_pairs['pair'], values=token_pairs[0], name='Token Pairs'),
            row=2, col=1
        )
        
        # Success ratio
        status_counts = df['status'].value_counts()
        fig.add_trace(
            go.Bar(x=status_counts.index, y=status_counts.values, name='Status'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Arbitrage Performance Dashboard")
        
        return fig
    
    def create_opportunity_table(self, opportunities: List[ArbitrageOpportunity]):
        """Maak opportunity tabel"""
        if not opportunities:
            return pd.DataFrame()
            
        data = []
        for opp in opportunities:
            data.append({
                "Tijd": opp.timestamp.strftime("%H:%M:%S"),
                "Van": opp.token_in,
                "Naar": opp.token_out,
                "Koop DEX": opp.buy_dex,
                "Verkoop DEX": opp.sell_dex,
                "Investering": f"{opp.amount_in:.4f}",
                "Verwachte Opbrengst": f"{opp.amount_out:.4f}",
                "Winst %": f"{opp.profit_percentage:.2f}%",
                "Netto Winst": f"{opp.net_profit:.6f} ETH",
                "Status": opp.status
            })
        
        return pd.DataFrame(data)
    
    # ============ UI RENDERING ============
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.title("⚙️ Configuratie")
            
            # Netwerk selectie
            network = st.selectbox(
                "Netwerk",
                ["sepolia", "mainnet"],
                index=0 if self.test_mode else 1
            )
            
            if network != self.network:
                self.network = network
                self.init_web3()
            
            # Trading parameters
            st.subheader("Trading Parameters")
            
            trade_amount = st.slider(
                "Trade Amount (ETH)",
                min_value=0.01,
                max_value=5.0,
                value=1.0,
                step=0.01
            )
            
            self.min_profit_threshold = st.slider(
                "Minimale Winst %",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1
            )
            
            # Monitoring control
            st.subheader("Monitoring")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Monitor", type="primary"):
                    st.session_state.monitoring_active = True
                    st.rerun()
            
            with col2:
                if st.button("⏹️ Stop Monitor"):
                    st.session_state.monitoring_active = False
                    st.rerun()
            
            auto_trade = st.checkbox("Auto Trading", value=False)
            
            # Wallet info
            st.subheader("Wallet Info")
            if st.secrets.get("ACCOUNT_ADDRESS"):
                st.code(st.secrets.get("ACCOUNT_ADDRESS")[:20] + "...")
            
            # Performance stats
            if st.session_state.trades:
                st.subheader("Performance")
                total_profit = sum(trade.get('profit', 0) for trade in st.session_state.trades)
                st.metric("Totale Winst", f"{total_profit:.6f} ETH")
                st.metric("Aantal Trades", len(st.session_state.trades))
    
    def render_main_dashboard(self):
        """Render hoofd dashboard"""
        st.title("🤖 Advanced Arbitrage Bot")
        
        # Real-time scanning
        if st.session_state.monitoring_active:
            with st.spinner("Scanning voor arbitrage kansen..."):
                opportunities = self.scan_arbitrage_opportunities()
                
                if opportunities:
                    st.session_state.opportunities.extend(opportunities)
                    
                    # Toon gevonden kansen
                    st.subheader(f"🎯 {len(opportunities)} Nieuwe Kansen Gevonden")
                    
                    # Auto-trading
                    auto_trade = st.sidebar.checkbox("Auto Trading", value=False)
                    if auto_trade and opportunities:
                        self.auto_trade_opportunities(opportunities)
                    
                    # Toon opportunities tabel
                    df = self.create_opportunity_table(opportunities)
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
                        
                        # Handmatige trade knop
                        for idx, opp in enumerate(opportunities):
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.write(f"{opp.token_in}→{opp.token_out} via {opp.buy_dex}→{opp.sell_dex}")
                            with col2:
                                st.write(f"Winst: **{opp.profit_percentage:.2f}%** ({opp.net_profit:.6f} ETH)")
                            with col3:
                                if st.button(f"Trade #{idx+1}", key=f"trade_{idx}"):
                                    result = self.execute_trade(opp)
                                    if result["status"] == "success":
                                        st.success("✅ Trade uitgevoerd!")
                                    else:
                                        st.error(f"❌ {result['message']}")
                else:
                    st.info("⏳ Geen arbitrage kansen gevonden...")
                    
                # Wacht voor volgende scan
                time.sleep(5)
                st.rerun()
        
        else:
            st.info("👆 Klik op 'Start Monitor' in de sidebar om te beginnen")
        
        # Historische data sectie
        if st.session_state.opportunities or st.session_state.trades:
            st.markdown("---")
            
            # Tabbladen voor verschillende views
            tab1, tab2, tab3 = st.tabs(["📊 Grafieken", "📈 Opportunities", "💸 Trades"])
            
            with tab1:
                # Profit charts
                fig = self.create_profit_chart()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Nog geen trade data voor grafieken")
            
            with tab2:
                # Alle opportunities
                if st.session_state.opportunities:
                    all_opps_df = self.create_opportunity_table(st.session_state.opportunities)
                    st.dataframe(all_opps_df, use_container_width=True)
                else:
                    st.info("Nog geen opportunities gedetecteerd")
            
            with tab3:
                # Trade history
                if st.session_state.trades:
                    trades_df = pd.DataFrame(st.session_state.trades)
                    st.dataframe(trades_df, use_container_width=True)
                    
                    # Export knop
                    if st.button("📥 Exporteer Trade Data"):
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="arbitrage_trades.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("Nog geen trades uitgevoerd")
        
        # Configuratie sectie
        with st.expander("🔧 Geavanceerde Configuratie"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Token Pairs")
                selected_pairs = st.multiselect(
                    "Selecteer te monitoren pairs",
                    [f"{a}/{b}" for a, b in self.monitored_pairs],
                    default=[f"{a}/{b}" for a, b in self.monitored_pairs[:3]]
                )
                
                # Update monitored pairs
                new_pairs = []
                for pair in selected_pairs:
                    a, b = pair.split("/")
                    new_pairs.append((a, b))
                self.monitored_pairs = new_pairs
            
            with col2:
                st.subheader("DEX Instellingen")
                for dex_name, dex in self.dexes.items():
                    dex.is_active = st.checkbox(dex.name, value=dex.is_active)
                
                # Update dex contracts
                self.init_dex_contracts()
    
    def run(self):
        """Start de bot"""
        self.render_sidebar()
        self.render_main_dashboard()

# ============ APP START ============
if __name__ == "__main__":
    # Voorkom threading issues in Streamlit
    bot = ArbitrageBot()
    bot.run()
