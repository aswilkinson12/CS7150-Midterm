#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read a URL list (one per line) exported from CyAN File Search and download all files.
Auth: either ~/.netrc (recommended) or appkey (?appkey=XXXX) appended to each URL.

Usage:
  python cyan_bulk_download.py urls.txt ./downloads  # .netrc 方式
  OBDAAC_APPKEY=your_key python cyan_bulk_download.py urls.txt ./downloads  # appkey 方式
"""
import os, sys, time, pathlib, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode

DEFAULT_WORKERS = 4
CHUNK = 131072  # 128KB, per OB.DAAC example
RETRIES = 4
SLEEP_BETWEEN = 0.05  # polite throttle

def add_appkey(url: str, key: str) -> str:
    if not key:
        return url
    parts = list(urlsplit(url))
    q = dict(parse_qsl(parts[3]))
    q["appkey"] = key
    parts[3] = urlencode(q)
    return urlunsplit(parts)

def filename_from_url_or_cd(resp, url):
    # try Content-Disposition filename
    cd = resp.headers.get("Content-Disposition")
    if cd and "filename=" in cd:
        fn = cd.split("filename=")[-1].strip().strip('"')
        return fn
    # fallback to URL path
    return pathlib.Path(urlparse(url).path).name or "download.dat"

def download_one(url, outdir: pathlib.Path, session: requests.Session, appkey: str):
    url2 = add_appkey(url, appkey)
    # final name before request (in case server provides Content-Disposition we may override later)
    tentative = pathlib.Path(urlparse(url2).path).name.split("?")[0]
    ofile = outdir / tentative
    if ofile.exists() and ofile.stat().st_size > 0:
        return str(ofile)

    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(url2, stream=True, timeout=300) as r:
                # Earthdata auth failure often returns an HTML login page
                if r.status_code != 200 or (
                    r.headers.get("Content-Type","").startswith("text/html")
                    and "<title>Earthdata Login</title>" in r.text[:4096]
                ):
                    raise RuntimeError(f"Auth/HTTP error: {r.status_code}")
                # if server tells us a better filename, switch
                realname = filename_from_url_or_cd(r, url2)
                ofile = outdir / realname
                tmp = ofile.with_suffix(ofile.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
                tmp.replace(ofile)
                time.sleep(SLEEP_BETWEEN)
                return str(ofile)
        except Exception as e:
            if attempt >= RETRIES:
                raise
            time.sleep(1.5 * attempt)

def main():
    if len(sys.argv) < 3:
        print("Usage: python cyan_bulk_download.py <urls.txt> <outdir> [workers]")
        sys.exit(1)
    urlfile = pathlib.Path(sys.argv[1])
    outdir = pathlib.Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_WORKERS
    appkey = os.environ.get("OBDAAC_APPKEY", "").strip()

    urls = [ln.strip() for ln in urlfile.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    print(f"Found {len(urls)} URLs. Saving to {outdir.resolve()}  (workers={workers})")

    with requests.Session() as sess, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(download_one, u, outdir, sess, appkey) for u in urls]
        ok = 0; fail = 0
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                p = fu.result(); ok += 1
                if i % 50 == 0:
                    print(f"Progress: {i}/{len(urls)}")
            except Exception as e:
                fail += 1
                print(f"[FAIL] {e}")
        print(f"Done. Success: {ok}, Fail: {fail}")

if __name__ == "__main__":
    main()