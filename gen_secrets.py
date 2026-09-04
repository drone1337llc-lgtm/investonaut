"""Generate .streamlit/secrets.toml on the LOCAL machine by reading the real
keys from cudacuda's .env. Never prints values. Writes the file directly.
Run on the local (teamamd) box:  python gen_secrets.py
"""
import subprocess, os, sys, tempfile

REMOTE = "surge@192.168.68.67"
ENV_PATH = "/home/surge/protonaut-live/cryptobot/.env"

# Pull the .env to a temp local file (never echo contents)
tmp = os.path.join(tempfile.gettempdir(), "_protonaut_env_pull")
subprocess.run(["scp", f"{REMOTE}:{ENV_PATH}", tmp], check=True)

env = {}
with open(tmp, encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

os.remove(tmp)

# Map cudacuda .env names -> Streamlit secrets names
mapping = {
    "ALPACA_API_KEY": "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY": "ALPACA_SECRET_KEY",
    "OLLAMA_CLOUD_URL": "OLLAMA_CLOUD_URL",
    "OLLAMA_CLOUD_KEY": "OLLAMA_CLOUD_KEY",
    "PROTONAUT_MODEL_BULL": "PROTONAUT_MODEL_BULL",
    "PROTONAUT_MODEL_BEAR": "PROTONAUT_MODEL_BEAR",
    "PROTONAUT_MODEL_MGR": "PROTONAUT_MODEL_MGR",
}

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "secrets.toml")

lines = ["# Auto-generated from cudacuda .env — DO NOT COMMIT (gitignored).\n"]
missing = []
for src, dst in mapping.items():
    if src in env and env[src]:
        # TOML string value; escape backslashes/quotes
        val = env[src].replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{dst} = "{val}"\n')
    else:
        missing.append(src)

with open(out_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Wrote {out_path}")
print("Keys written:", [mapping[s] for s in mapping if s in env and env[s]])
if missing:
    print("MISSING from cudacuda .env:", missing)
