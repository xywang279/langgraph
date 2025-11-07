import logging
from pathlib import Path

from . import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("app.log", encoding="utf-8"), logging.StreamHandler()]
)

storage.init_db()

DOC_ROOT = Path(__file__).resolve().parent.parent / "docs"
DOC_ROOT.mkdir(parents=True, exist_ok=True)
