# code/main.py

"""
+-----------------------------------------------------------------------------+
|                                CLAIM ANALYZER     HACKERRANK ORCHESTRATE CHALLENGE                          |
+-----------------------------------------------------------------------------+
+-----------------------------------+
|   |
+-----------------------------------+

@link https://www.hackerrank.com/contests/hackerrank-orchestrate-june26/challenges/multi-modal-review
Clone Repo - @link https://github.com/jasonmonroe/multi-modal-evidence-review-system

Build a system that verifies visual evidence for damage claims across three
object types: cars, laptops, and packages.

Your system will receive claim conversations, one or more submitted images,
user claim history, and minimum evidence requirements. It must decide whether
the submitted images support the claim, contradict it, or do not provide enough
information.
"""

__author__ = "Jason Monroe (jason@jasonmonroe.com)"
__copyright__ = "Copyright © 2011-2026"
__date__ = "2026-07-08"
__version__ = "1.1.0"

# Python Libraries
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing anything that reads MODEL_* from the environment
# (src.constants does so at import time). The path is anchored to this file so
# it works from any working directory.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Local Libraries
from evaluation.main import run_evaluation_report_pipeline
from pipelines.run_claims_pipeline import run_claims_pipeline
from src.data_handler import DataHandler
from src.image_handler import ImageHandler
from src.utils import show_banner, show_timer, start_timer


def run_main_pipeline(args: dict):
    subtitles = ["The Insurance Agent Adjuster at your service!"]
    for key, value in args.items():
        icon = "✅" if value else "❌"
        subtitles.append(f"Args: {key.title().replace('_', ' ').lower()}: {icon} ")

    show_banner("HackerRank Orchestrate".upper(), subtitles)

    # Process data
    data_handle = DataHandler()
    data_handle.load_data(args)
    image_handle = ImageHandler(args.get("eda", False))

    if args.get("eda"):
        data_handle.describe()

    run_claims_pipeline(data_handle, image_handle)

    # Run evaluation pipeline
    if args.get("eval"):
        run_evaluation_report_pipeline(data_handle, image_handle)


def parse_args(command_line_args: list[str]) -> dict:
    args_list = ["--eda", "--eval", "--sample"]
    return {arg.strip("--"): (arg in command_line_args) for arg in args_list}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    prog_start_time = start_timer()
    run_id = str(int(prog_start_time))[-6:]
    print(f"\n----- 🖨️️ START RUN ID: {run_id} 🖨️️ -----")

    args = parse_args(sys.argv[1:])
    run_main_pipeline(args)

    show_timer(prog_start_time)
    print(f"\n----- 🖨️️ END RUN ID: {run_id} 🖨️️ -----\n")
