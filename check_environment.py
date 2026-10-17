from importlib.util import find_spec
import platform
import sys

MODULES = [
    "dotenv",
    "openai",
    "pydantic",
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "sentence_transformers",
    "faiss",
    "rank_bm25",
    "fastapi",
    "uvicorn",
    "httpx",
    "pytest",
    "google.genai"
]

missing = [name for name in MODULES if find_spec(name) is None]

print("Python:", sys.version.split()[0])
print("System:", platform.platform())

if find_spec("torch"):
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA dostępna:", torch.cuda.is_available())

if missing:
    print("Brakujące moduły:", ", ".join(missing))
    raise SystemExit(1)

print("Środowisko jest gotowe.")