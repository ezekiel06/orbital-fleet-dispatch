"""Step 9a: Download and decode real small-satellite telemetry from SatNOGS.

SatNOGS is a volunteer network of ground stations that record satellite radio
signals. This pulls recent raw frames for one satellite and decodes them with
SatNOGS's own decoders into readings like battery voltage and temperatures.

Needs a free SatNOGS DB account: put your API token (from db.satnogs.org,
Settings / API Token) in satnogs_token.txt next to this script. Never share it.

Usage:  pip install satnogs-decoders
        python satnogs_fetch.py [--norad 68635] [--decoder Canvas] [--pages 40]
        python satnogs_fetch.py --hours 24 --pages 300 --resume   # slow full-day pull
Saves:  satnogs_<norad>.jsonl   one decoded frame per line
"""
import argparse, json, time, urllib.request
import satnogsdecoders.decoder as dec
from satnogsdecoders.decoder import get_fields

API = "https://db.satnogs.org/api/telemetry/?satellite={norad}&format=json"


def fetch(norad, pages, token, hours=None, state_path=None, on_page=None):
    """Pages back in time. Slow on purpose: SatNOGS is volunteer-run and limits speed.
    Backs off when told "too many requests", and can resume from a saved page."""
    import os, urllib.error
    from datetime import datetime, timezone, timedelta
    url, rows = API.format(norad=norad), []
    if state_path and os.path.exists(state_path):
        url = open(state_path).read().strip() or url
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours) if hours else None
    for n in range(pages):
        req = urllib.request.Request(url, headers={"Authorization": f"Token {token}",
                                                   "User-Agent": "sat-sim-research"})
        wait = 15
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 5:
                    wait = int(e.headers.get("Retry-After") or wait * 2)
                    print(f"  rate limited, waiting {wait}s", flush=True)
                    time.sleep(wait); continue
                print(f"stopping early: {e}"); return rows
            except Exception as e:
                if attempt == 5:
                    print(f"stopping early: {e}"); return rows
                time.sleep(10)
        rows += d["results"]
        if on_page:
            on_page(d["results"])  # save each page right away, so an interrupted run loses nothing
        url = d.get("next")
        if state_path:
            open(state_path, "w").write(url or "")
        oldest = d["results"][-1]["timestamp"] if d["results"] else None
        print(f"  page {n + 1}: {len(rows)} frames, back to {oldest}", flush=True)
        if not url or (cutoff and oldest and datetime.fromisoformat(oldest.replace("Z", "+00:00")) < cutoff):
            break
        time.sleep(3)  # be polite to a volunteer-run service
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norad", default="68635")
    ap.add_argument("--decoder", default="Canvas")
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--hours", type=float, help="stop once data is this many hours old")
    ap.add_argument("--resume", action="store_true", help="continue from the last saved page, append")
    a = ap.parse_args()
    token = open("satnogs_token.txt").read().strip()
    state = f"satnogs_{a.norad}.next" if a.resume else None
    D = getattr(dec, a.decoder)
    path = f"satnogs_{a.norad}.jsonl"
    out = open(path, "a" if a.resume else "w")
    count = {"ok": 0}

    def save(page):
        for r in page:
            try:
                fields = get_fields(D.from_bytes(bytes.fromhex(r["frame"])))
            except Exception:
                continue
            if fields:
                count["ok"] += 1
                out.write(json.dumps({"timestamp": r["timestamp"], "station": r["observer"], **fields}, default=str) + "\n")
        out.flush()

    rows = fetch(a.norad, a.pages, token, a.hours, state, on_page=save)
    out.close()
    print(f"{len(rows)} frames downloaded, {count['ok']} decoded -> {path}")


if __name__ == "__main__":
    main()
