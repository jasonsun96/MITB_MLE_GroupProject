import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCLUDE_DIR = PROJECT_ROOT / "include"
for path in (PROJECT_ROOT, INCLUDE_DIR):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def main() -> None:
    if "--predict-only" in sys.argv:
        from model_pipeline.model_inference import main as run_inference

        run_inference()
    else:
        from model_pipeline.model_training import main as run_training

        run_training()


if __name__ == "__main__":
    main()
