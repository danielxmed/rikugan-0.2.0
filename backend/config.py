import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8080"))
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/root/.cache/huggingface")
DEVICE: str = os.getenv("DEVICE", "cuda")
DTYPE: str = os.getenv("DTYPE", "bfloat16")
MAX_SEQ_LEN: int = int(os.getenv("MAX_SEQ_LEN", "512"))
