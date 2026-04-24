import os
import gc
import numpy as np
import polars as pl
import scipy.sparse as sp
import implicit
import faiss
from pathlib import Path
from typing import List, Dict, Protocol, Optional
from dataclasses import dataclass

from interfaces import BatchContext


class FeatureSource(Protocol):
    @property
    def name(self) -> str:
        """ """

    def fit(self, train_data: pl.LazyFrame):
        """ """

    def startup(self):
        """ """

    def shutdown(self):
        """ """

    def transform(self, candidates: pl.LazyFrame, context: BatchContext) -> pl.LazyFrame:
        """ """

class FeatureManager:
    def __init__(self, sources: List[FeatureSource]):
        self.sources = sources

    def fit_all(self, train_data: pl.LazyFrame):
        for src in self.sources:
            print(f'      Fit now: {src.name}')
            src.fit(train_data)

    def startup(self):
        for src in self.sources:
            src.startup()

    def shutdown(self):
        for src in self.sources:
            src.shutdown()

    def extract(self, candidates: pl.LazyFrame, context: BatchContext) -> pl.LazyFrame:
        """Последовательно применяет все источники фичей к кандидатам"""
        out = candidates
        for src in self.sources:
            print(f'      Extract now: {src.name}')
            out = src.transform(out, context)
        return out


class ItemStaticFeatureSource:
    def __init__(self, artist_mapping: pl.DataFrame):
        self.artist_mapping = artist_mapping.lazy()
        self.path = Path("fittingdata/features/item_static")
        self.path.mkdir(parents=True, exist_ok=True)
        self.data = None

    @property
    def name(self) -> str: return "item_static"

    def fit(self, train_data: pl.LazyFrame):
        print(f"[{self.name}] Fitting...")
        # Считаем глобальные счетчики трека
        stats = train_data.group_by("item_id").agg([
            pl.len().alias("item_cnt_likes"),
            pl.col("is_organic").mean().alias("item_organic_ratio")
        ])
        # Приклеиваем метаданные (артиста)
        stats = stats.join(self.artist_mapping, on="item_id", how="left").collect()
        stats.write_parquet(self.path / "item_features.parquet")

    def startup(self):
        self.data = pl.scan_parquet(self.path / "item_features.parquet")

    def shutdown(self):
        self.data = None
        gc.collect()

    def transform(self, candidates: pl.LazyFrame, context: BatchContext) -> pl.LazyFrame:
        return candidates.join(self.data, on="item_id", how="left").fill_null(0)


class UserStaticFeatureSource:
    def __init__(self):
        self.path = Path("fittingdata/features/user_static")
        self.path.mkdir(parents=True, exist_ok=True)
        self.data = None

    @property
    def name(self) -> str: return "user_static"

    def fit(self, train_data: pl.LazyFrame):
        print(f"[{self.name}] Fitting...")
        stats = train_data.group_by("uid").agg([
            pl.len().alias("user_total_likes"),
            pl.col("item_id").n_unique().alias("user_unique_items")
        ]).collect()
        stats.write_parquet(self.path / "user_features.parquet")

    def startup(self):
        self.data = pl.scan_parquet(self.path / "user_features.parquet")

    def shutdown(self):
        self.data = None

    def transform(self, candidates: pl.LazyFrame, context: BatchContext) -> pl.LazyFrame:
        # ОПТИМИЗАЦИЯ: Джойним фичи только для юзеров в текущем батче
        batch_users = context.target_users.select("uid")
        relevant_stats = self.data.join(batch_users, on="uid", how="inner")
        return candidates.join(relevant_stats, on="uid", how="left").fill_null(0)


class IALSDotProductSource:
    def __init__(self, ials_cg):
        self.cg = ials_cg

    @property
    def name(self) -> str: return "ials_dot_product"

    def fit(self, train_data: pl.LazyFrame): pass

    def startup(self):
        # Больше никаких словарей!
        pass 

    def shutdown(self):
        pass

    def transform(self, candidates: pl.LazyFrame, context: BatchContext) -> pl.LazyFrame:
        # 1. Приклеиваем индексы u_idx и i_idx. 
        # Т.к. self.cg.user_map_df теперь физический, джойн отработает мгновенно и надежно.
        enriched = (
            candidates
            .join(self.cg.user_map_df.lazy(), on="uid", how="left")
            .join(self.cg.item_map_df.lazy(), on="item_id", how="left")
        )

        # 2. Обертка для математики
        def calculate_dot(df: pl.DataFrame) -> pl.Series:
            # Если юзера или айтема нет в iALS, индексы будут null
            u_rows = df["u_idx"].to_numpy()
            i_rows = df["i_idx"].to_numpy()

            # Создаем маску валидных индексов (заменяем null на -1)
            mask = (np.nan_to_num(u_rows, nan=-1) >= 0) & (np.nan_to_num(i_rows, nan=-1) >= 0)
            dots = np.zeros(len(df), dtype=np.float32)

            if mask.any():
                # Берем только целые части индексов для numpy
                u_idx = u_rows[mask].astype(np.int32)
                i_idx = i_rows[mask].astype(np.int32)
                
                u_vecs = self.cg.user_factors[u_idx]
                i_vecs = self.cg.item_factors[i_idx]
                dots[mask] = np.einsum('ij,ij->i', u_vecs, i_vecs)

            return pl.Series("als_dot", dots, dtype=pl.Float32)

        # 3. Используем select + with_columns для максимальной стабильности
        # Мы передаем в функцию только 2 колонки, а результат приклеиваем обратно
        return enriched.with_columns([
            pl.struct(["u_idx", "i_idx"])
            .map_batches(lambda s: calculate_dot(s.struct.unnest()))
            .alias("als_dot")
        ]).with_columns(
            # Чем выше скор, тем меньше ранг (1 - лучший трек)
            pl.col("als_dot").rank(descending=True, method="min").over("uid").alias("als_rank")
        ).drop(["u_idx", "i_idx"])


class ItemTrendFeatureSource:
    def __init__(self, days_trend=14):
        self.days_trend = days_trend
        self.path = Path("fittingdata/features/item_trend")
        self.path.mkdir(parents=True, exist_ok=True)
        self.data = None

    @property
    def name(self) -> str: return "item_trend"

    def fit(self, train_data: pl.LazyFrame):
        print(f"[{self.name}] Fitting trend stats...")
        
        # Находим самый последний timestamp в тренировочных данных
        max_ts = train_data.select(pl.col("timestamp").max()).collect().item()
        
        # Вычисляем порог: 14 дней в секундах (зависит от размерности timestamp в датасете, 
        # предполагаем, что там секунды или миллисекунды. Для секунд:)
        threshold = max_ts - (self.days_trend * 24 * 60 * 60)

        # Считаем статистику
        stats = train_data.group_by("item_id").agg([
            pl.len().alias("total_likes_for_trend"),
            pl.col("timestamp").filter(pl.col("timestamp") > threshold).len().alias(f"item_likes_last_{self.days_trend}d")
        ]).with_columns(
            # Доля свежих лайков (чем ближе к 1, тем трек "горячее")
            (pl.col(f"item_likes_last_{self.days_trend}d") / pl.col("total_likes_for_trend")).alias("item_trend_ratio")
        ).drop("total_likes_for_trend").collect() # total_likes уже есть в ItemStatic

        stats.write_parquet(self.path / "item_trend.parquet")

    def startup(self):
        self.data = pl.scan_parquet(self.path / "item_trend.parquet")

    def shutdown(self):
        self.data = None

    def transform(self, candidates: pl.LazyFrame, context: BatchContext) -> pl.LazyFrame:
        return candidates.join(self.data, on="item_id", how="left").fill_null(0.0)
    


class UserArtistAffinityFeatureSource:
    def __init__(self, artist_mapping: pl.DataFrame):
        self.artist_mapping = artist_mapping.select([
            pl.col("item_id").cast(pl.UInt32),
            pl.col("artist_id").fill_null(0).cast(pl.UInt32)
        ]).lazy()

    @property
    def name(self) -> str: return "user_artist_affinity"
    def fit(self, train_data: pl.LazyFrame): pass
    def startup(self): pass
    def shutdown(self): pass # ИСПРАВЛЕНО: Теперь не упадет

    def transform(self, candidates: pl.LazyFrame, context: BatchContext) -> pl.LazyFrame:
        # Принудительно UInt32 для джойнов
        cand = candidates.with_columns([pl.col("uid").cast(pl.UInt32), pl.col("item_id").cast(pl.UInt32)])
        hist = context.history_likes.with_columns([pl.col("uid").cast(pl.UInt32), pl.col("item_id").cast(pl.UInt32)])

        batch_history = (
            hist.join(context.target_users.select(pl.col("uid").cast(pl.UInt32)), on="uid", how="inner")
            .join(self.artist_mapping, on="item_id", how="inner")
        )
        
        user_artist_counts = batch_history.group_by(["uid", "artist_id"]).agg([
            pl.len().alias("user_artist_likes_cnt")
        ]).with_columns(pl.col("artist_id").cast(pl.UInt32))

        return (
            cand
            .join(self.artist_mapping, on="item_id", how="left")
            .with_columns(pl.col("artist_id").fill_null(0).cast(pl.UInt32)) # ИСПРАВЛЕНО: защита от f64
            .join(user_artist_counts, on=["uid", "artist_id"], how="left")
            .with_columns(pl.col("user_artist_likes_cnt").fill_null(0))
            .drop("artist_id")
        )

class ArtistStaticFeatureSource:
    def __init__(self, artist_mapping: pl.DataFrame):
        self.artist_mapping = artist_mapping.select([
            pl.col("item_id").cast(pl.UInt32),
            pl.col("artist_id").fill_null(0).cast(pl.UInt32)
        ]).lazy()
        self.path = Path("fittingdata/features/artist_static")
        self.path.mkdir(parents=True, exist_ok=True)
        self.data = None

    @property
    def name(self) -> str: return "artist_static"

    def fit(self, train_data: pl.LazyFrame):
        print(f"[{self.name}] Fitting artist stats...")
        df = train_data.join(self.artist_mapping, on="item_id", how="inner")
        stats = df.group_by("artist_id").agg([
            pl.len().alias("artist_total_likes"),
            pl.col("item_id").n_unique().alias("artist_unique_liked_tracks")
        ]).with_columns(pl.col("artist_id").cast(pl.UInt32)).collect()
        stats.write_parquet(self.path / "artist_features.parquet")

    def startup(self):
        self.data = pl.scan_parquet(self.path / "artist_features.parquet").with_columns(
            pl.col("artist_id").cast(pl.UInt32)
        )

    def shutdown(self): # ИСПРАВЛЕНО: Теперь не упадет
        self.data = None
        gc.collect()

    def transform(self, candidates: pl.LazyFrame, context: BatchContext) -> pl.LazyFrame:
        cand_with_art = candidates.join(self.artist_mapping, on="item_id", how="left").with_columns(
            pl.col("artist_id").fill_null(0).cast(pl.UInt32)
        )
        return cand_with_art.join(self.data, on="artist_id", how="left").drop("artist_id").fill_null(0)
    
    
class CandidateSourceFeatureSource:
    """Считает, из скольких источников пришел кандидат (is_ials + is_pop + ...)"""
    
    @property
    def name(self) -> str: return "candidate_source_stats"

    def fit(self, train_data: pl.LazyFrame): pass
    def startup(self): pass
    def shutdown(self): pass

    def transform(self, candidates: pl.LazyFrame, context: BatchContext) -> pl.LazyFrame:
        # Находим все колонки, начинающиеся на is_
        is_cols = [c for c in candidates.columns if c.startswith("is_")]
        
        if not is_cols:
            return candidates
            
        # Используем sum_horizontal для быстрого сложения колонок в Polars
        return candidates.with_columns(
            pl.sum_horizontal(is_cols).alias("source_count")
        )

# --- В ячейке инициализации FeatureManager добавь его в список ---
# item_stats = ItemStaticFeatureSource(...)
# user_stats = UserStaticFeatureSource()
# source_stats = CandidateSourceFeatureSource() # <--- НОВЫЙ ИСТОЧНИК
# feature_manager = FeatureManager(sources=[item_stats, user_stats, ials_dot, source_stats])