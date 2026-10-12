"""Batch audit entry point for V14; no scientific experiment is repeated."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from tools import tracking_audit

if __name__ == "__main__":
    tracking_audit.ROOT = ROOT
    tracking_audit.main()
