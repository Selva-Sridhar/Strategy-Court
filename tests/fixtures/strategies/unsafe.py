"""Unsafe: network, OS access, data fetching and sandbox-escape attempts. Never executed."""

import os
import subprocess

import pandas as pd
import requests
import yfinance as yf


def signals(prices: pd.DataFrame) -> pd.DataFrame:
    extra = yf.download("SPY")
    requests.get("http://example.com")
    os.system("whoami")
    subprocess.run(["ls"])
    open("/etc/passwd").read()
    eval("1 + 1")
    ().__class__.__bases__[0].__subclasses__()
    return prices * 0 + len(extra)
