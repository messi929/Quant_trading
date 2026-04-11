"""KIS API broker wrapper — reuses V2 KISApi.

V2 KISApi is stable and tested with real orders.
This module re-exports it for V3 with a clean interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to import V2 broker
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from broker.kis_api import KISApi

__all__ = ["KISApi"]
