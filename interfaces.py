from dataclasses import dataclass
from typing import List, Dict, Protocol

import numpy as np
import polars as pl


@dataclass
class BatchContext:
    target_users: pl.LazyFrame

    history_likes: pl.LazyFrame
    history_listens: pl.LazyFrame

    user_embeddings: np.ndarray = None
    user_id_to_idx: Dict[int, int] = None
