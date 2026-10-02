"""Compatibility patches for external dependencies (e.g. PEFT/torchao version mismatches)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_torchao_compat() -> None:
    """Prevent PEFT from crashing when an older torchao version is present in the environment."""
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass

    try:
        import peft.tuners.lora.torchao
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass
