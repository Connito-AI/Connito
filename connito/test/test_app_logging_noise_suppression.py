"""Verify configure_logging() filters the noisy huggingface_hub permissions warning.

The chmod-after-move race in `huggingface_hub.file_download` emits two
multi-line warnings per fetched file (one for the file, one for its repeat in
the cache subtree). On a busy validator that's tens of thousands of lines of
pure noise per day. `configure_logging` installs a filter that drops only
those specific messages while leaving the rest of huggingface_hub's warnings
intact.
"""
from __future__ import annotations

import logging

from connito.shared.app_logging import configure_logging


def test_huggingface_hub_permissions_warning_is_suppressed(caplog):
    configure_logging()
    hf_logger = logging.getLogger("huggingface_hub.file_download")

    with caplog.at_level(logging.WARNING, logger="huggingface_hub.file_download"):
        hf_logger.warning(
            "Could not set the permissions on the file '/tmp/x.incomplete'. "
            "Error: [Errno 2] No such file or directory.\n"
            "Continuing without setting permissions."
        )
        hf_logger.warning("Some other huggingface_hub warning the user should still see")

    messages = [r.getMessage() for r in caplog.records]
    assert not any("Could not set the permissions" in m for m in messages), (
        "Noisy chmod-race warning should be filtered out"
    )
    assert any("Some other huggingface_hub warning" in m for m in messages), (
        "Unrelated huggingface_hub warnings must still surface"
    )
