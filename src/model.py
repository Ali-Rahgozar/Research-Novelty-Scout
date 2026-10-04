import json
from pathlib import Path

from openai import OpenAI



CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

API_KEY = config["api_key"]
BASE_URL = config["base_url"]
MODEL_NAME = config["model"]

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)
