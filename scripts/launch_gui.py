"""Web GUI Launcher for Google Colab and Local Execution.

Launches the interactive Gradio Blocks dashboard with share=True
to generate a public temporary URL accessible outside Google Colab.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_forecast.web.app import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("launch_gui")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Crypto Forecast Gradio GUI")
    parser.add_argument("--share", action="store_true", default=True, help="Enable Gradio public link (share=True)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server host IP")
    args = parser.parse_args()

    logger.info("Initializing Gradio UI...")
    demo = create_app()

    logger.info(f"Starting server on {args.server_name}:{args.port} (share={args.share})...")
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
