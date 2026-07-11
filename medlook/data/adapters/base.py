"""Common intermediate record and adapter interface shared by every data adapter.

Each adapter (`open_vqa`, `meissa`, `uncertainty`) converts its own raw source format
into a stream of `Record` objects. Everything downstream (`convert.py`, `filter.py`,
`decontaminate.py`, `curriculum.py`) only ever deals with `Record` -- never with
adapter-specific raw formats -- which keeps the adapters swappable and testable in
isolation, per the "pluggable, not Meissa-locked" design goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from PIL import Image

from medlook.schema import Strategy


@dataclass
class Record:
    """One training example, before it is rendered into the MedLook text schema and
    the Unsloth chat-message format (that happens in `convert.py`)."""

    id: str
    images: List[Image.Image]
    question: str
    strategy: Strategy
    final_answer: str
    final_confidence: float
    process: Optional[str] = None
    source: str = "unknown"
    difficulty: str = "unknown"
    gold: bool = False
    extra: dict = field(default_factory=dict)


class Adapter:
    """Base interface every data adapter implements."""

    name: str = "base"

    def is_available(self) -> bool:
        """Return True if this adapter's data source is reachable/usable.

        Adapters that depend on an optional network download (e.g. `meissa`) must
        override this and return False gracefully rather than raising, so
        `prepare_data.py` can continue in a degraded-but-functional mode instead of
        hard-failing on an optional source.
        """
        return True

    def iter_records(self) -> Iterator[Record]:
        raise NotImplementedError
