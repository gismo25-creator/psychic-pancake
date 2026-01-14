import csv
import time
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

BASE = "https://mempool.space"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "btc-fragment-search/1.1"})

def get_json(url, retries=6, backoff=0.7):
    for i in range(retries):
        r = SESSION.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 503, 502):
            time.sleep(backoff * (2 ** i))
            continue
        raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:200]}")
    raise RuntimeError(f"Failed after retries: {url}")

def ams_to_utc_ts(year, month, day, hour, minute):
    dt_local = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("Europe/Amsterdam"))
    return int(dt_local.astimezone(timezone.utc).timestamp())

def sats_from_btc_str(btc_str: str) -> int:
    whole, _, frac = btc_str.partition(".")
    frac = (frac + "0" * 8)[:8]
    return int(whole) * 100_000_000 + int(frac)

def iso_utc(ts: int | None):
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def normalize_block_item(b):
    """
    Accepts either:
      - dict with keys like id/hash/height/timestamp
      - string with block hash/id
    Returns: (block_hash, height, timestamp)
    """
    if isinstance(b, dict):
        block_hash = b.get("id") or b.get("hash")
        height = b.get("height")
        ts = b.get("timestamp")
        return block_hash, height, ts

    if isinstance(b, str):
        block_hash = b.strip()
        # Fetch full block info so we can log height/timestamp consistently
        info = get_json(f"{BASE}/api/block/{block_hash}")
        # mempool block info typically has: id, height, timestamp
        height = info.get("height")
        ts = info.get("timestamp")
        return block_hash, height, ts

    raise TypeError(f"Unexpected block item type: {type(b)}")

def scan_block(block_hash, block_height, block_ts, prefix, suffix, target_sats,
               tol_sats=0, include_candidates=True, sleep_s=0.1):
    """
    Returns:
      matches: exact (addr prefix+suffix AND value == target_sats)
      candidates: addr prefix+suffix AND abs(value-target_sats) <= tol_sats  (if enabled)
    """
    start = 0
    matches = []
    candidates = []

    while True:
        txs = get_json(f"{BASE}/api/block/{block_hash}/txs/{start}")
        if not txs:
            break

        for tx in txs:
            txid = tx.get("txid")
            status = tx.get("status", {})
            tx_block_time = status.get("block_time")  # unix secs (confirmed)
            vouts = tx.get("vout", [])

            for vout_index, vout in enumerate(vouts):
                addr = vout.get("scriptpubkey_address") or ""
                val = vout.get("value")
                if val is None or not addr:
                    continue

                if addr.startswith(prefix) and addr.endswith(suffix):
                    row = {
                        "block_height": block_height,
                        "block_hash": block_hash,
                        "block_time_utc": iso_utc(block_ts),
                        "txid": txid,
                        "tx_block_time_utc": iso_utc(tx_block_time),
                        "vout_index": vout_index,
                        "address": addr,
                        "value_sats": val,
                        "value_btc": f"{val/100_000_000:.8f}",
                        "target_sats": target_sats,
                        "target_btc": f"{target_sats/100_000_000:.8f}",
                        "delta_sats": val - target_sats,
                    }

                    if val == target_sats:
                        matches.append(row)
                    elif include_candidates and abs(val - target_sats) <= tol_sats:
                        candidates.append(row)

        start += len(txs)
        time.sleep(sleep_s)

    return matches, candidates

def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def main():
    # jouw inputs
    prefix = "bc1q7t4"
    suffix = "h0m78fz"
    target_btc = "0.19789241"
    target_sats = sats_from_btc_str(target_btc)

    # tijd: 14 jan 2026 19:40 Europe/Amsterdam
    ts = ams_to_utc_ts(2026, 1, 14, 19, 40)
    print("UTC:", iso_utc(ts), "timestamp:", ts)

    # haal blokken rond timestamp
    blocks = get_json(f"{BASE}/api/v1/mining/blocks/timestamp/{ts}")
    print("Blocks returned:", len(blocks))

    # instellingen
    TOL_SATS = 200  # zet op 0 als je alleen exact wilt; 100-500 is vaak genoeg
    INCLUDE_CANDIDATES = True

    all_matches = []
    all_candidates = []

    for b in blocks:
    block_hash, height, bts = normalize_block_item(b)
    print(f"Scanning height={height} hash={block_hash} ...")

    matches, candidates = scan_block(
        block_hash=block_hash,
        block_height=height,
        block_ts=bts,
        prefix=prefix,
        suffix=suffix,
        target_sats=target_sats,
        tol_sats=TOL_SATS,
        include_candidates=INCLUDE_CANDIDATES
    )
    all_matches.extend(matches)
    all_candidates.extend(candidates)

    # schrijf CSVs
    write_csv("btc_matches.csv", all_matches)
    write_csv("btc_candidates.csv", all_candidates)

    print(f"Exact matches: {len(all_matches)} -> btc_matches.csv")
    print(f"Candidates (tol={TOL_SATS} sats): {len(all_candidates)} -> btc_candidates.csv")
    if not all_matches:
        print("Geen exacte match gevonden in deze blokset.")
        print("Als je tijd niet 100% exact is: probeer ts ± 3600 (1 uur) door timestamp aan te passen,")
        print("of scan een groter venster (zie tip hieronder).")

if __name__ == "__main__":
    main()
