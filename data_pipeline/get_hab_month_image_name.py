import re
import datetime as dt
from pathlib import Path

ROOT = Path("./data")   # Data directory
FILES = sorted(list(ROOT.glob("*.tif")))

# Daily： ... LYYYYDDD.L3m_DAY_ ... _col_row.tif
PAT_DAY  = re.compile(r".*?[A-Z](\d{4})(\d{3})\.L3m_DAY_.*?_(\d+)_(\d+)\.tif$", re.IGNORECASE)

def doy_to_date(year:int, doy:int) -> dt.date:
    return dt.date(year, 1, 1) + dt.timedelta(days=doy-1)

def parse_date_from_name(name:str):
    m = PAT_DAY.match(name)
    if m:
        y, doy = int(m.group(1)), int(m.group(2))
        return doy_to_date(y, doy)
    return None

selected = []
for f in FILES:
    d = parse_date_from_name(f.name)
    if d and 6 <= d.month <= 11:
        selected.append((d, f))


selected.sort(key=lambda x: x[0])
print(f"Found {len(selected)} files in months 6–10:")
for d, f in selected[:10]:
    print(d.isoformat(), "->", f.name)

with open("jun_to_oct_files.txt", "w") as w:
    for d, f in selected:
        w.write(f"{d.isoformat()}\t{f}\n")
