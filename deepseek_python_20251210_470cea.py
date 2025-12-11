# Ethereum Arbitrage Bot - Veilige Versie
import os
import time
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from web3 import Web3
from dotenv import load_dotenv
from datetime import datetime
import requests
from decimal import Decimal

# Laad environment variables uit veilige bronnen
load_dotenv()

# ============ VEILIGE CONFIGURATIE ============
# Gebruik Streamlit secrets of environment variables
# NOOIT hardcoded gevoelige data!

def load_secrets():
    """Laad gevoelige data veilig"""
    try:
        # Eerst Streamlit secrets proberen
        secrets = st.secrets
        
        # Controleer of we in testmodus zijn
        TEST_MODE = secrets.get("TEST_MODE", "True") == "True"
        
        config = {
            "TEST_MODE": TEST_MODE,
            "PRIVATE_KEY": secrets.get("PRIVATE_KEY", ""),
            "ACCOUNT_ADDRESS": secrets.get("ACCOUNT_ADDRESS", ""),
            "RPC_URL": secrets.get("RPC_URL", ""),
            "ETHERSCAN_API": secrets.get("ETHERSCAN_API", ""),
            "MAX_LOSS_ETH": float(secrets.get("MAX_LOSS_ETH", 0.1)),
            "MAX_TRADE_ETH": float(secrets.get("MAX_TRADE_ETH", 0.5))
        }
        
        # Voor testnet, gebruik Goerli of Sepolia
        if config["TEST_MODE"]:
            config["RPC_URL"] = "https://eth-sepolia.g.alchemy.com/v2/BA0gd1_jd7ZZNaWU_5lQJ"  # Demo URL
            st.warning("⚠️ TEST MODE ACTIEF - Gebruik alleen testnet!")
        
        return config
    except Exception as e:
        st.error(f"Fout bij laden configuratie: {e}")
        return None

# Laad configuratie
config = load_secrets()
if not config:
    st.error("Configuratie niet geladen. Controleer je .env of secrets.toml")
    st.stop()

# ============ TESTNET INSTELLINGEN ============
if config["TEST_MODE"]:
    # Testnet contract adressen
    UNISWAP_ROUTER = Web3.to_checksum_address("0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD")  # Sepolia Uniswap
    SUSHISWAP_ROUTER = Web3.to_checksum_address("0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506")  # Sepolia Sushi
    
    # Testnet tokens
    TOKENS = {
        "WETH": Web3.to_checksum_address("0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"),  # Sepolia WETH
        "DAI": Web3.to_checksum_address("0x11fE4B6AE13d2a6055C8D9cF65c55bac32B5d844"),   # Sepolia DAI
        "USDC": Web3.to_checksum_address("0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"), # Sepolia USDC
        "TEST_ETH": "ETH"
    }
    
    # Testnet URLs
    ETHERSCAN_URL = "https://sepolia.etherscan.io"
else:
    # Mainnet contract adressen (alleen gebruiken als TEST_MODE=False)
    UNISWAP_ROUTER = Web3.to_checksum_address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")
    SUSHISWAP_ROUTER = Web3.to_checksum_address("0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F")
    
    # Mainnet tokens
    TOKENS = {
        "WETH": Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"),
        "DAI": Web3.to_checksum_address("0x6B175474E89094C44Da98b954EedeAC495271d0F"),
        "USDC": Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606EB48"),
        "USDT": Web3.to_checksum_address("0xdAC17F958D2ee523a2206206994597C13D831ec7"),
        "ETH": "ETH"
    }
    
    ETHERSCAN_URL = "https://etherscan.io"

# ============ INITIALISATIE ============
@st.cache_resource
def init_web3():
    """Initialiseer Web3 met caching"""
    try:
        w3 = Web3(Web3.HTTPProvider(config["RPC_URL"]))
        if w3.is_connected():
            return w3
        else:
            st.error("Kan niet verbinden met blockchain")
            return None
    except Exception as e:
        st.error(f"Fout bij initialiseren Web3: {e}")
        return None

w3 = init_web3()
if not w3:
    st.stop()

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

# ============ VEILIGE HELPER FUNCTIES ============
def calculate_safe_gas_price():
    """Bereken veilige gas price met buffer"""
    try:
        current_gas = w3.eth.gas_price
        # Voeg 20% buffer toe
        safe_gas = int(current_gas * 1.2)
        # Maximum van 100 Gwei
        max_gas = w3.to_wei(100, 'gwei')
        return min(safe_gas, max_gas)
    except:
        return w3.to_wei(30, 'gwei')  # Fallback

def check_token_balance(token_address, account_address):
    """Controleer token balans"""
    if token_address == "ETH":
        return w3.eth.get_balance(account_address)
    else:
        # ERC20 balance check (vereist token ABI)
        try:
            token_abi = [{"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"}]
            token_contract = w3.eth.contract(address=token_address, abi=token_abi)
            return token_contract.functions.balanceOf(account_address).call()
        except:
            return 0

def simulate_transaction(tx):
    """Simuleer transactie vooraf (indien mogelijk)"""
    try:
        # Probeer eth_call om transactie te simuleren
        result = w3.eth.call(tx)
        return True, "Simulatie geslaagd"
    except Exception as e:
        return False, f"Simulatie mislukt: {str(e)}"

# ============ VEILIGE ARBITRAGE FUNCTIE ============
def check_arbitrage_opportunity(tokenA, tokenB, amount_wei):
    """Veilige arbitrage check"""
    try:
        # Valideer inputs
        if tokenA not in TOKENS or tokenB not in TOKENS:
            return None, "Ongeldige tokens"
        
        if amount_wei <= 0:
            return None, "Bedrag moet positief zijn"
        
        tokenA_addr = TOKENS[tokenA] if tokenA != "ETH" else TOKENS["WETH"]
        tokenB_addr = TOKENS[tokenB] if tokenB != "ETH" else TOKENS["WETH"]
        
        path = [tokenA_addr, tokenB_addr]
        
        # Controleer liquiditeit
        try:
            uni_amounts = uni_router.functions.getAmountsOut(amount_wei, path).call()
            sushi_amounts = sushi_router.functions.getAmountsOut(amount_wei, path).call()
        except Exception as e:
            return None, f"Liquiditeitsfout: {str(e)}"
        
        if len(uni_amounts) < 2 or len(sushi_amounts) < 2:
            return None, "Onvoldoende liquiditeit"
        
        uni_out = uni_amounts[-1]
        sushi_out = sushi_amounts[-1]
        
        # Bereken winst
        profit_wei = sushi_out - uni_out if sushi_out > uni_out else uni_out - sushi_out
        profit_eth = w3.from_wei(profit_wei, 'ether')
        
        # Bereken percentages
        if uni_out > 0:
            profit_percent = (profit_wei / uni_out) * 100
        else:
            profit_percent = 0
        
        # Schat gas kosten (in wei)
        gas_limit = 250000
        gas_price = calculate_safe_gas_price()
        estimated_gas_cost = gas_limit * gas_price
        
        # Netto winst (na gas kosten)
        net_profit_wei = profit_wei - estimated_gas_cost
        net_profit_eth = w3.from_wei(net_profit_wei, 'ether')
        
        return {
            "tokenA": tokenA,
            "tokenB": tokenB,
            "amount_eth": w3.from_wei(amount_wei, 'ether'),
            "uni_out_eth": w3.from_wei(uni_out, 'ether'),
            "sushi_out_eth": w3.from_wei(sushi_out, 'ether'),
            "profit_eth": profit_eth,
            "profit_percent": profit_percent,
            "gas_cost_eth": w3.from_wei(estimated_gas_cost, 'ether'),
            "net_profit_eth": net_profit_eth,
            "net_profit_percent": (net_profit_wei / amount_wei * 100) if amount_wei > 0 else 0,
            "timestamp": datetime.now(),
            "direction": "Sushi->Uni" if sushi_out > uni_out else "Uni->Sushi"
        }, None
        
    except Exception as e:
        return None, f"Fout bij arbitrage check: {str(e)}"

# ============ VEILIGE TRADE UITVOERING ============
def execute_safe_trade(trade_info, slippage_percent=0.5, confirm=True):
    """Veilige trade uitvoering met meerdere checks"""
    
    # Controleer of we in testmodus zijn
    if config["TEST_MODE"]:
        st.warning("⚠️ TEST MODE - Geen echte trades worden uitgevoerd")
        return {"status": "test", "message": "Test mode actief"}
    
    # Controleer configuratie
    if not config.get("PRIVATE_KEY") or not config.get("ACCOUNT_ADDRESS"):
        return {"status": "error", "message": "Wallet niet geconfigureerd"}
    
    # Valideer trade info
    required_keys = ["tokenA", "tokenB", "amount_eth", "direction"]
    for key in required_keys:
        if key not in trade_info:
            return {"status": "error", "message": f"Ontbrekende trade info: {key}"}
    
    # Controleer maximum trade limiet
    if trade_info["amount_eth"] > config["MAX_TRADE_ETH"]:
        return {"status": "error", "message": f"Trade bedrag overschrijdt maximum ({config['MAX_TRADE_ETH']} ETH)"}
    
    # Vraag handmatige bevestiging
    if confirm:
        col1, col2, col3 = st.columns(3)
        with col1:
            if not st.button("✅ Bevestig Trade", key="confirm_trade"):
                return {"status": "cancelled", "message": "Trade geannuleerd door gebruiker"}
    
    try:
        # Bereken amount in wei
        amount_wei = w3.to_wei(trade_info["amount_eth"], 'ether')
        
        # Bereken path
        token_in = TOKENS[trade_info["tokenA"]] if trade_info["tokenA"] != "ETH" else TOKENS["WETH"]
        token_out = TOKENS[trade_info["tokenB"]] if trade_info["tokenB"] != "ETH" else TOKENS["WETH"]
        path = [token_in, token_out]
        
        # Kies router op basis van direction
        if trade_info["direction"] == "Uni->Sushi":
            router = uni_router
        else:
            router = sushi_router
        
        # Schat output amount
        amounts_out = router.functions.getAmountsOut(amount_wei, path).call()
        expected_out = amounts_out[-1]
        
        # Bereken minimum output met slippage
        min_amount_out = int(expected_out * (1 - slippage_percent / 100))
        
        # Bereken deadline (10 minuten)
        deadline = int(time.time()) + 600
        
        # Bouw transactie
        nonce = w3.eth.get_transaction_count(config["ACCOUNT_ADDRESS"])
        gas_price = calculate_safe_gas_price()
        
        # Als we ETH sturen
        if trade_info["tokenA"] == "ETH":
            tx = router.functions.swapExactETHForTokens(
                min_amount_out,
                path,
                config["ACCOUNT_ADDRESS"],
                deadline
            ).build_transaction({
                'from': config["ACCOUNT_ADDRESS"],
                'value': amount_wei,
                'gas': 250000,
                'gasPrice': gas_price,
                'nonce': nonce,
                'chainId': 1  # Mainnet
            })
        else:
            # Voor token->token swaps (vereist approve eerst)
            return {"status": "error", "message": "Token->Token swaps nog niet geïmplementeerd"}
        
        # Simuleer transactie eerst
        simulation_ok, sim_message = simulate_transaction(tx)
        if not simulation_ok:
            return {"status": "error", "message": f"Simulatie mislukt: {sim_message}"}
        
        # Onderteken en verstuur
        signed_tx = w3.eth.account.sign_transaction(tx, config["PRIVATE_KEY"])
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wacht op bevestiging
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        
        if receipt.status == 1:
            return {
                "status": "success",
                "message": f"Trade succesvol!",
                "tx_hash": tx_hash.hex(),
                "etherscan_url": f"{ETHERSCAN_URL}/tx/{tx_hash.hex()}"
            }
        else:
            return {"status": "error", "message": "Transactie mislukt op blockchain"}
            
    except Exception as e:
        return {"status": "error", "message": f"Trade fout: {str(e)}"}

# ============ STREAMLIT UI ============
st.set_page_config(page_title="Veilige Arbitrage Bot", layout="wide")
st.title("🛡️ Veilige Ethereum Arbitrage Bot")

# Sidebar configuratie
with st.sidebar:
    st.header("⚙️ Configuratie")
    
    # Netwerk info
    st.subheader("Netwerk Status")
    if w3.is_connected():
        current_block = w3.eth.block_number
        st.success(f"✅ Verbonden (Block: {current_block})")
        st.info(f"Test Mode: {'✅ ACTIEF' if config['TEST_MODE'] else '❌ Uit'}")
    else:
        st.error("❌ Niet verbonden")
    
    # Trading limieten
    st.subheader("Trading Limieten")
    max_trade = st.number_input("Max trade (ETH)", 
                                min_value=0.01, 
                                max_value=10.0, 
                                value=config["MAX_TRADE_ETH"])
    
    min_profit_percent = st.number_input("Minimale winst (%)", 
                                        min_value=0.01, 
                                        max_value=5.0, 
                                        value=0.5)
    
    slippage = st.number_input("Slippage (%)", 
                              min_value=0.1, 
                              max_value=5.0, 
                              value=0.5)
    
    # Token selectie
    st.subheader("Token Pairs")
    token_options = list(TOKENS.keys())
    token_pairs = []
    
    col1, col2 = st.columns(2)
    with col1:
        tokenA = st.selectbox("Token A", token_options, index=0)
    with col2:
        tokenB = st.selectbox("Token B", token_options, index=1)
    
    if st.button("Voeg pair toe"):
        if (tokenA, tokenB) not in token_pairs:
            token_pairs.append((tokenA, tokenB))
            st.success(f"Pair toegevoegd: {tokenA}/{tokenB}")

# Hoofdpagina
st.header("🔍 Arbitrage Scanner")

# Trade amount
col1, col2 = st.columns(2)
with col1:
    trade_amount = st.number_input("Trade amount (ETH)", 
                                  min_value=0.01, 
                                  max_value=max_trade, 
                                  value=0.1)
with col2:
    scan_interval = st.number_input("Scan interval (seconden)", 
                                   min_value=5, 
                                   max_value=60, 
                                   value=10)

# Controleer wallet balans (veilige manier zonder privésleutel te gebruiken)
if config["ACCOUNT_ADDRESS"]:
    try:
        balance_wei = w3.eth.get_balance(config["ACCOUNT_ADDRESS"])
        balance_eth = w3.from_wei(balance_wei, 'ether')
        st.info(f"💰 Wallet balans: {balance_eth:.4f} ETH")
    except:
        st.warning("Kan wallet balans niet ophalen")

# Start/stop controls
col1, col2 = st.columns(2)
with col1:
    start_scan = st.button("🚀 Start Scan", type="primary")
with col2:
    stop_scan = st.button("⏹️ Stop Scan")

# Logging
if 'log_data' not in st.session_state:
    st.session_state.log_data = []

if 'scanning' not in st.session_state:
    st.session_state.scanning = False

if start_scan:
    st.session_state.scanning = True
    st.success("Scanning gestart!")

if stop_scan:
    st.session_state.scanning = False
    st.warning("Scanning gestopt!")

# Scan loop
if st.session_state.scanning:
    placeholder = st.empty()
    
    with placeholder.container():
        st.info("Scanning voor arbitrage kansen...")
        
        # Controleer alle token pairs
        for tokenA, tokenB in token_pairs:
            if tokenA == tokenB:
                continue
                
            result, error = check_arbitrage_opportunity(
                tokenA, 
                tokenB, 
                w3.to_wei(trade_amount, 'ether')
            )
            
            if result and error is None:
                # Check of winst hoog genoeg is
                if result["net_profit_percent"] >= min_profit_percent:
                    # Toon arbitrage kans
                    st.success(f"💸 Arbitrage kans gevonden!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=f"{result['tokenA']} → {result['tokenB']}",
                            value=f"{result['net_profit_percent']:.2f}%",
                            delta=f"{result['net_profit_eth']:.6f} ETH"
                        )
                    
                    with col2:
                        st.write(f"**Route:** {result['direction']}")
                        st.write(f"**Gas kosten:** {result['gas_cost_eth']:.6f} ETH")
                        st.write(f"**Netto winst:** {result['net_profit_eth']:.6f} ETH")
                    
                    # Voeg toe aan log
                    st.session_state.log_data.append(result)
                    
                    # Trade knop (met bevestiging)
                    if st.button(f"Trade {result['tokenA']}→{result['tokenB']}", 
                                key=f"trade_{result['tokenA']}_{result['tokenB']}"):
                        # Vraag extra bevestiging
                        with st.expander("⚠️ Trade bevestiging"):
                            st.write(f"Weet je zeker dat je deze trade wilt uitvoeren?")
                            st.write(f"**Amount:** {trade_amount} ETH")
                            st.write(f"**Verwachte winst:** {result['net_profit_eth']:.6f} ETH")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("✅ Ja, voer trade uit", key="confirm_final"):
                                    trade_result = execute_safe_trade(
                                        {
                                            "tokenA": result["tokenA"],
                                            "tokenB": result["tokenB"],
                                            "amount_eth": trade_amount,
                                            "direction": result["direction"]
                                        },
                                        slippage_percent=slippage,
                                        confirm=False  # We hebben al bevestigd
                                    )
                                    
                                    if trade_result["status"] == "success":
                                        st.success(trade_result["message"])
                                        if "etherscan_url" in trade_result:
                                            st.markdown(f"[Bekijk transactie]({trade_result['etherscan_url']})")
                                    else:
                                        st.error(trade_result["message"])
                            
                            with col2:
                                if st.button("❌ Annuleer", key="cancel_final"):
                                    st.warning("Trade geannuleerd")
            
            elif error:
                st.warning(f"{tokenA}→{tokenB}: {error}")
        
        # Wacht voor volgende scan
        time.sleep(scan_interval)
        st.rerun()

# Toon historische data
if st.session_state.log_data:
    st.header("📊 Historische Data")
    
    df = pd.DataFrame(st.session_state.log_data)
    
    # Toon tabel
    st.dataframe(df.sort_values("timestamp", ascending=False))
    
    # Toon grafieken
    col1, col2 = st.columns(2)
    
    with col1:
        if len(df) > 1:
            fig1 = px.line(df, x="timestamp", y="net_profit_percent", 
                          color="tokenA", title="Netto Winst %")
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if len(df) > 1:
            fig2 = px.bar(df, x="timestamp", y="net_profit_eth", 
                         color="direction", title="Netto Winst (ETH)")
            st.plotly_chart(fig2, use_container_width=True)

# Instructies
with st.expander("📚 Veiligheidsinstructies"):
    st.markdown("""
    ### 🔒 VEILIGHEIDSMAATREGELEN:
    
    1. **ALTIJD TESTNET EERST** - Gebruik altijd testnet (Sepolia/Goerli) voordat je mainnet gebruikt
    2. **KLEINE BEDRAGEN** - Begin met kleine bedragen (max 0.1 ETH)
    3. **HANDMATIGE BEVESTIGING** - Elke trade vereist handmatige bevestiging
    4. **SLIPPAGE INSTELLEN** - Zet slippage op minimaal 0.5%
    5. **WALLET VEILIGHEID**:
       - Gebruik een aparte wallet voor trading
       - Zet nooit al je funds in één wallet
       - Gebruik hardware wallet indien mogelijk
    
    ### ⚠️ RISICO'S:
    - **Impermanent Loss**: Liquidity pools kunnen waarde verliezen
    - **Slippage**: Prijs kan veranderen tijdens transactie
    - **Gas Kosten**: Transacties kunnen duur zijn
    - **Smart Contract Risico**: Bugs in contracten kunnen funds locken
    
    ### 🚨 NOOIT DOEN:
    - Nooit je mainnet privésleutel in code zetten
    - Nooit auto-trading aan zonder limieten
    - Nooit meer inleggen dan je kan verliezen
    """)

# Footer
st.markdown("---")
st.caption("⚠️ Deze tool is voor educatieve doeleinden. Gebruik op eigen risico.")
