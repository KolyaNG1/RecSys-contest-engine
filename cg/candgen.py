from typing import Protocol, List, Dict

import os
from pathlib import Path
import pickle
import gc

import numpy as np
import polars as pl
import scipy.sparse as sp
import implicit
import faiss


from interfaces import BatchContext


class CandidateGeneratorInterface(Protocol):
    @property
    def name(self) -> str:
        """ get name """

    def startup(self):
        """Загружаем из диска всю стату"""

    def shutdown(self):
        """выгружаем все из ОЗУ на диск"""

    def generate(self, context: BatchContext, num_candidates: int) -> pl.DataFrame:
        """
        Должен вернуть ОДНУ большую таблицу для ВСЕХ юзеров из context.uids.
        Колонки: [uid, item_id, score, source]
        """

    def fit(self, train_data: pl.LazyFrame):
        """hui"""


class RetrievalStage:
    def __init__(self, generators: List['CandidateGeneratorInterface']):
        self.generators = generators

    def fetch_all(self, context: 'BatchContext', num_candidates_per_gen: int) -> pl.LazyFrame:
        dataframes = []
        for gen in self.generators:
            print(f'      Generate now: {gen.name}')
            res = gen.generate(context, num_candidates_per_gen)
            res_lf = res.lazy() if isinstance(res, pl.DataFrame) else res
            
            # Добавляем колонку-флаг для конкретного генератора сразу
            # Это создаст колонку типа is_ials = 1.0
            res_lf = res_lf.with_columns(pl.lit(1.0).alias(f"from_{gen.name}"))
            
            dataframes.append(res_lf)

        combined_lf = pl.concat(dataframes, how="diagonal")

        # Группируем, чтобы собрать все флаги в одну строку
        # И агрегируем скоры (например, берем максимум)
        return (
            combined_lf
            .group_by(["uid", "item_id"])
            .agg([
                # Берем максимальный скор (или сумму)
                pl.col("score").max().alias("score"),
                # Собираем флаги (если хоть один генератор дал 1, будет 1, иначе null -> 0)
                *[pl.col(f"from_{gen.name}").max().fill_null(0.0).alias(f"is_{gen.name}") 
                  for gen in self.generators]
            ])
            # Убираем то, что уже лайкано
            .join(context.history_likes, on=["uid", "item_id"], how="anti")
            .cast({"uid": pl.UInt32, "item_id": pl.UInt32})
        )


class IALSCandidateGenerator:
    def __init__(self, factors=128, iterations=30, alpha=40, regularization=0.1, file_prefix=None):
        self.factors = factors
        self.iterations = iterations
        self.alpha = alpha
        self.regularization = regularization

        pr = file_prefix if file_prefix else ''
        self.base_path = Path(f"fittingdata/cg/{pr + '_' + self.name}")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Объекты в RAM
        self.user_factors = None
        self.item_factors = None
        self.hnsw_index = None
        self.user_id_map = None
        self.item_id_map = None
        self.uid_to_row = None

    @property
    def name(self) -> str:
        return "ials"

    def _is_fitted(self) -> bool:
        return (self.base_path / "index.bin").exists() and (self.base_path / "item_factors.npy").exists()

    
    def fit(self, train_data: pl.LazyFrame, force: bool = False):
        if not force and self._is_fitted():
            print(f"[{self.name}] УЖЕ ОБУЧЕН! Нашел кэш в {self.base_path}. Пропускаю fit().")
            return

        print(f"[{self.name}] Fitting model...")

        df = train_data.select(["uid", "item_id"]).collect()

        # 1. Маппинги
        user_map = df.select("uid").unique().get_column("uid").to_numpy().astype(np.uint32)
        item_map = df.select("item_id").unique().get_column("item_id").to_numpy().astype(np.uint32)

        df_users = pl.DataFrame({"uid": user_map}).with_row_index("u_idx")
        df_items = pl.DataFrame({"item_id": item_map}).with_row_index("i_idx")

        df_mapped = (
            df.join(df_users, on="uid", how="left")
              .join(df_items, on="item_id", how="left")
        )

        u_idx = df_mapped["u_idx"].to_numpy()
        i_idx = df_mapped["i_idx"].to_numpy()

        # 2. Матрица и обучение
        counts = np.ones(len(u_idx), dtype=np.float32) * self.alpha
        matrix = sp.csr_matrix((counts, (u_idx, i_idx)), shape=(len(user_map), len(item_map)))

        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors, 
            iterations=self.iterations, 
            regularization=self.regularization, # Сильно помогает при больших factors
            random_state=42,
            use_gpu=False # На CPU 80M строк пережуются отлично
        )
        model.fit(matrix)

        # === ФИКС ДЛЯ GPU: ПЕРЕНОС В RAM ===
        # Выгружаем факторы из видеокарты в обычную память
        u_factors = model.user_factors
        i_factors = model.item_factors

        if hasattr(u_factors, "to_numpy"):
            u_factors = u_factors.to_numpy()
        if hasattr(i_factors, "to_numpy"):
            i_factors = i_factors.to_numpy()

        # 3. Сохраняем маппинги и факторы (используем u_factors/i_factors!)
        pl.DataFrame({"uid": user_map}).write_parquet(self.base_path / "user_map.parquet")
        pl.DataFrame({"item_id": item_map}).write_parquet(self.base_path / "item_map.parquet")
        
        np.save(self.base_path / "user_factors.npy", u_factors)
        np.save(self.base_path / "item_factors.npy", i_factors)

        # 4. Строим Faiss индекс (используем i_factors, которые уже в RAM)
        # ВАЖНО: берем i_factors, а не model.item_factors
        item_vectors = np.ascontiguousarray(i_factors, dtype=np.float32)
        
        # index = faiss.IndexHNSWFlat(item_vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexFlatIP(item_vectors.shape[1])
        index.add(item_vectors)
        faiss.write_index(index, str(self.base_path / "index.bin"))

        print(f"[{self.name}] Fit complete. Data moved to CPU and saved to {self.base_path}")

    def generate(self, context: 'BatchContext', num_candidates: int) -> pl.DataFrame:
        batch_with_idx = (
            context.target_users
            .join(self.user_map_df.lazy(), on="uid", how="inner") # <--- ДОБАВЛЕНО .lazy()
            .collect()
        )
        
        uids = batch_with_idx["uid"].to_numpy()
        u_indices = batch_with_idx["u_idx"].to_numpy()

        if len(uids) == 0: return self._empty_df()

        # Faiss Search
        scores, indices = self.hnsw_index.search(self.user_factors[u_indices], num_candidates)

        return pl.DataFrame({
            "uid": np.repeat(uids, num_candidates),
            "item_id": self.item_id_map_raw[indices.flatten()],
            "score": scores.flatten()
        }).with_columns(pl.lit(self.name).alias("source")).cast({"uid": pl.UInt32, "item_id": pl.UInt32})


    def startup(self):
        if not self._is_fitted():
            raise RuntimeError(f"[{self.name}] Model not fitted!")

        print(f"[{self.name}] Loading from disk (Stable version)...")
        # 1. Загружаем маппинги как физические DataFrame (не Lazy!)
        # Это гарантирует, что колонки u_idx и i_idx всегда будут существовать
        user_ids = pl.read_parquet(self.base_path / "user_map.parquet")["uid"].cast(pl.UInt32)
        self.user_map_df = pl.DataFrame({
            "uid": user_ids,
            "u_idx": np.arange(len(user_ids), dtype=np.int32)
        })

        item_ids = pl.read_parquet(self.base_path / "item_map.parquet")["item_id"].cast(pl.UInt32)
        self.item_map_df = pl.DataFrame({
            "item_id": item_ids,
            "i_idx": np.arange(len(item_ids), dtype=np.int32)
        })

        # Загружаем вектора и индекс (mmap)
        self.user_factors = np.load(self.base_path / "user_factors.npy") # Без mmap!
        self.item_factors = np.load(self.base_path / "item_factors.npy") # Без mmap!


        # был index.bin - это старый файлик от faiss.IndexFlatIP(item_vectors.shape[1])
        self.hnsw_index = faiss.read_index(str(self.base_path / "index_hnsw_flat.bin"), faiss.IO_FLAG_MMAP)
        
        # Сохраняем словарь для быстрого поиска в методе generate
        self.uid_to_row = {uid: i for i, uid in enumerate(user_ids.to_numpy())}
        self.item_id_map_raw = item_ids.to_numpy()

    def shutdown(self):
        self.user_factors = self.item_factors = self.hnsw_index = None
        self.user_map_df = self.item_map_df = self.uid_to_row = None
        gc.collect()

    def _empty_df(self):
        return pl.DataFrame(schema={"uid": pl.UInt32, "item_id": pl.UInt32, "score": pl.Float32, "source": pl.Utf8})


class IALSItemToItemGenerator:
    def __init__(self, n_last_items=15, n_similar_per_item=30, file_prefix=None):
        self.n_last_items = n_last_items
        self.n_similar_per_item = n_similar_per_item
        
        pr = file_prefix if file_prefix else ''
        # Используем ту же папку, что и обычный iALS, чтобы не дублировать файлы
        self.base_path = Path(f"fittingdata/cg/{pr + '_ials'}") 
        
        # Объекты в RAM
        self.item_factors = None
        self.hnsw_index = None
        self.item_map_df = None
        self.item_id_map_raw = None
        self.item_to_idx_dict = None # Для мгновенного маппинга истории

    @property
    def name(self) -> str:
        return "ials_i2i"

    def _is_fitted(self) -> bool:
        return (self.base_path / "index_hnsw_flat.bin").exists()

    def fit(self, train_data: pl.LazyFrame, force: bool = False):
        # Этот генератор использует артефакты обычного IALSCandidateGenerator.
        # Если тот обучен, этот тоже "обучен".
        if not self._is_fitted():
            print(f"[{self.name}] Error: Базовый iALS не найден. Сначала обучи обычный IALSCandidateGenerator.")
        pass

    def startup(self):
        if not self._is_fitted():
            raise RuntimeError(f"[{self.name}] Base IALS artifacts not found!")

        print(f"[{self.name}] Starting up i2i engine...")
        
        # 1. Загружаем маппинг айтемов
        item_ids = pl.read_parquet(self.base_path / "item_map.parquet")["item_id"].cast(pl.UInt32)
        self.item_id_map_raw = item_ids.to_numpy()
        
        # Создаем словарь для быстрого получения индекса вектора по item_id
        # Это в разы быстрее, чем Join в Polars для маленьких списков истории
        self.item_to_idx_dict = {iid: idx for idx, iid in enumerate(self.item_id_map_raw)}

        # 2. Загружаем факторы и HNSW индекс
        self.item_factors = np.load(self.base_path / "item_factors.npy")
        self.hnsw_index = faiss.read_index(str(self.base_path / "index_hnsw_flat.bin"), faiss.IO_FLAG_MMAP)

    def generate(self, context: 'BatchContext', num_candidates: int) -> pl.DataFrame:
        # 1. Извлекаем историю только для текущего батча пользователей
        # В Polars Lazy лучше делать join вместо is_in для фильтрации по списку
        # context.history и context.target_users — это LazyFrames
        history_df = (
            context.history_likes
            .join(context.target_users.select("uid"), on="uid", how="inner")
            .sort(["uid", "timestamp"], descending=[False, True])
            .group_by("uid")
            .head(self.n_last_items)
            .collect() # Превращаем в DataFrame, чтобы работать с numpy
        )

        if history_df.height == 0:
            return self._empty_df()

        # 2. Теперь history_df — это DataFrame, используем .get_column()
        uids_list = history_df.get_column("uid").to_numpy()
        item_ids_list = history_df.get_column("item_id").to_numpy()
        
        valid_uids = []
        valid_indices = []

        # Быстрый маппинг item_id -> i_idx через словарь
        for i in range(len(item_ids_list)):
            idx = self.item_to_idx_dict.get(item_ids_list[i])
            if idx is not None:
                valid_uids.append(uids_list[i])
                valid_indices.append(idx)

        if not valid_indices:
            return self._empty_df()

        # 3. Batch Search в Faiss
        query_vectors = self.item_factors[valid_indices]
        scores, indices = self.hnsw_index.search(query_vectors, self.n_similar_per_item)

        # 4. Собираем результат через Numpy
        uids_repeated = np.repeat(valid_uids, self.n_similar_per_item)
        items_found = self.item_id_map_raw[indices.flatten()]
        scores_found = scores.flatten()

        # 5. Оборачиваем в Polars, дедуплицируем и берем Top-N
        return (
            pl.DataFrame({
                "uid": uids_repeated,
                "item_id": items_found,
                "score": scores_found
            })
            .unique(subset=["uid", "item_id"])
            .sort(["uid", "score"], descending=[False, True])
            .group_by("uid")
            .head(num_candidates)
            .with_columns(pl.lit(self.name).alias("source"))
            .cast({"uid": pl.UInt32, "item_id": pl.UInt32})
        )

    def shutdown(self):
        self.item_factors = self.hnsw_index = None
        self.item_id_map_raw = self.item_to_idx_dict = None
        gc.collect()

    def _empty_df(self):
        return pl.DataFrame(schema={"uid": pl.UInt32, "item_id": pl.UInt32, "score": pl.Float32, "source": pl.Utf8})


# === 2. Global Popularity Генератор ===
class GlobalPopularityGenerator:
    def __init__(self, top_n=100):
        self.top_n = top_n
        self.base_path = Path(f"fittingdata/cg/{self.name}")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.top_items_df = None

    @property
    def name(self) -> str:
        return "global_pop"

    def fit(self, train_data: pl.LazyFrame):
        print(f"[{self.name}] Calculating global top...")
        top_df = (
            train_data.group_by("item_id").len()
            .sort("len", descending=True).head(self.top_n)
            .select([pl.col("item_id").cast(pl.UInt32), pl.col("len").cast(pl.Float32).alias("score")])
            .collect()
        )
        top_df.write_parquet(self.base_path / "top_items.parquet")
        print(f"[{self.name}] Saved to {self.base_path}")

    def startup(self):
        file = self.base_path / "top_items.parquet"
        if not file.exists(): raise RuntimeError(f"[{self.name}] Not fitted!")
        self.top_items_df = pl.read_parquet(file)

    def shutdown(self):
        self.top_items_df = None
        gc.collect()

    def generate(self, context: 'BatchContext', num_candidates: int) -> pl.DataFrame:
        uids = context.target_users.select("uid").collect()
        return uids.join(self.top_items_df, how="cross").with_columns(pl.lit(self.name).alias("source"))


# === 3. Artist Popularity Генератор ===
class ArtistPopularityGenerator:
    def __init__(self, artist_mapping: pl.DataFrame, top_n_artists=10, tracks_per_artist=10):
        self.artist_mapping_raw = artist_mapping
        self.top_n_artists = top_n_artists
        self.tracks_per_artist = tracks_per_artist

        self.base_path = Path(f"fittingdata/cg/{self.name}")
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.artist_top_tracks = None
        self.artist_mapping = None

    @property
    def name(self) -> str:
        return "artist_pop"

    def fit(self, train_data: pl.LazyFrame):
        print(f"[{self.name}] Precalculating artist top tracks...")
        # Сохраняем маппинг артистов для инференса
        self.artist_mapping_raw.write_parquet(self.base_path / "artist_mapping.parquet")

        # Считаем золотой фонд треков артистов
        top_tracks = (
            train_data
            .join(self.artist_mapping_raw.lazy(), on="item_id")
            .group_by(["artist_id", "item_id"]).len()
            .with_columns(pl.col("len").rank(descending=True).over("artist_id").alias("rank"))
            .filter(pl.col("rank") <= self.tracks_per_artist)
            .select(["artist_id", "item_id", pl.col("len").cast(pl.Float32).alias("artist_score")])
            .collect()
        )
        top_tracks.write_parquet(self.base_path / "artist_top_tracks.parquet")
        print(f"[{self.name}] Fit complete.")

    def startup(self):
        if not (self.base_path / "artist_top_tracks.parquet").exists():
            raise RuntimeError(f"[{self.name}] Not fitted!")
        self.artist_top_tracks = pl.read_parquet(self.base_path / "artist_top_tracks.parquet").lazy()
        self.artist_mapping = pl.read_parquet(self.base_path / "artist_mapping.parquet").lazy()

    def shutdown(self):
        self.artist_top_tracks = None
        self.artist_mapping = None
        gc.collect()

    def generate(self, context: 'BatchContext', num_candidates: int) -> pl.DataFrame:
        batch_uids = context.target_users
        user_fav_artists = (
            context.history_likes.join(batch_uids, on="uid")
            .join(self.artist_mapping, on="item_id")
            .group_by(["uid", "artist_id"]).len()
            .with_columns(pl.col("len").rank(descending=True).over("uid").alias("art_rank"))
            .filter(pl.col("art_rank") <= self.top_n_artists)
        )
        return (
            user_fav_artists.join(self.artist_top_tracks, on="artist_id")
            .select([pl.col("uid"), pl.col("item_id"), pl.col("artist_score").alias("score"),
                     pl.lit(self.name).alias("source")])
        ).collect()


import os
import gc
from pathlib import Path
import numpy as np
import polars as pl
import scipy.sparse as sp

import os
import gc
from pathlib import Path
import numpy as np
import polars as pl
import scipy.sparse as sp

class CoVisitationGenerator:
    def __init__(self, top_k_similar=100, history_depth=30, file_prefix=None):
        self.top_k = top_k_similar
        self.history_depth = history_depth
        
        pr = file_prefix if file_prefix else ''
        self.base_path = Path(f"fittingdata/cg/{pr + '_' + self.name}")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.S_matrix = None
        self.item_id_map_raw = None
        self.item_to_idx_dict = None

    @property
    def name(self) -> str:
        return "covisitation"
    
    def _is_fitted(self) -> bool:
        return (self.base_path / "ease_weights.npy").exists() and (self.base_path / "item_map.parquet").exists()

    def fit(self, train_data: pl.LazyFrame, force: bool = False):
        # 2. ДОБАВИТЬ ЭТИ 3 СТРОЧКИ В НАЧАЛО:
        if not force and self._is_fitted():
            print(f"[{self.name}] УЖЕ ОБУЧЕН! Нашел кэш в {self.base_path}. Пропускаю fit().")
            return
        
        print(f"[{self.name}] Building Sparse Matrix...")
        df = train_data.select(["uid", "item_id"]).collect()
        
        users = df["uid"].unique().to_numpy()
        items = df["item_id"].unique().to_numpy()
        
        pl.DataFrame({"item_id": items}).write_parquet(self.base_path / "item_map.parquet")
        
        # Индексируем на лету
        df = df.join(pl.DataFrame({"uid": users}).with_row_index("u_idx"), on="uid")
        df = df.join(pl.DataFrame({"item_id": items}).with_row_index("i_idx"), on="item_id")
        
        u_idx = df["u_idx"].to_numpy()
        i_idx = df["i_idx"].to_numpy()
        
        del df
        gc.collect()

        vals = np.ones(len(u_idx), dtype=np.float32)
        X = sp.csr_matrix((vals, (u_idx, i_idx)), shape=(len(users), len(items)))

        print(f"[{self.name}] Computing X^T * X (Item-Item Graph)...")
        G = X.T.dot(X)
        G.setdiag(0)
        
        del X
        gc.collect()

        print(f"[{self.name}] Sparsifying graph (keeping top {self.top_k} per row)...")
        # Оптимизированный код для обрезки хвостов CSR матрицы
        data, indices, indptr = [], [], [0]
        
        for i in range(G.shape[0]):
            start, end = G.indptr[i], G.indptr[i+1]
            if start < end:
                row_data = G.data[start:end]
                row_indices = G.indices[start:end]
                
                if len(row_data) > self.top_k:
                    # Быстрый поиск топ K
                    top_idx = np.argpartition(-row_data, self.top_k - 1)[:self.top_k]
                    row_data = row_data[top_idx]
                    row_indices = row_indices[top_idx]
                    
                data.extend(row_data)
                indices.extend(row_indices)
            indptr.append(len(data))
            
        # Создаем "чистую" усеченную матрицу
        S = sp.csr_matrix((data, indices, indptr), shape=G.shape)
        
        # Сохраняем в оптимизированном бинарном формате scipy
        sp.save_npz(self.base_path / "similarity_matrix.npz", S)
        print(f"[{self.name}] Fit complete!")

    def startup(self):
        file_path = self.base_path / "similarity_matrix.npz"
        if not file_path.exists(): raise RuntimeError(f"[{self.name}] Not fitted!")
        
        print(f"[{self.name}] Loading Sparse Similarity Matrix to RAM...")
        self.S_matrix = sp.load_npz(file_path)
        
        items = pl.read_parquet(self.base_path / "item_map.parquet")["item_id"].to_numpy()
        self.item_id_map_raw = items
        self.item_to_idx_dict = {iid: idx for idx, iid in enumerate(items)}

    def shutdown(self):
        self.S_matrix = None
        self.item_id_map_raw = None
        self.item_to_idx_dict = None
        gc.collect()

    def generate(self, context: 'BatchContext', num_candidates: int) -> pl.DataFrame:
        # 1. Извлекаем историю батча
        history = (
            context.history_likes
            .join(context.target_users, on="uid", how="inner")
            .sort(["uid", "timestamp"], descending=[False, True])
            .group_by("uid").head(self.history_depth)
            .select(["uid", "item_id"])
            .collect()
        )
        if history.height == 0: return self._empty_df()

        batch_uids = context.target_users.select("uid").collect().get_column("uid").to_numpy()
        uid_to_idx = {uid: idx for idx, uid in enumerate(batch_uids)}
        
        valid_u_idx, valid_i_idx = [], []
        for u, i in zip(history["uid"].to_numpy(), history["item_id"].to_numpy()):
            if i in self.item_to_idx_dict:
                valid_u_idx.append(uid_to_idx[u])
                valid_i_idx.append(self.item_to_idx_dict[i])
                
        if not valid_u_idx: return self._empty_df()

        # 2. Матрица истории (Батч Юзеров x Все Айтемы)
        X_batch = sp.csr_matrix(
            (np.ones(len(valid_u_idx), dtype=np.float32), (valid_u_idx, valid_i_idx)), 
            shape=(len(batch_uids), len(self.item_id_map_raw))
        )
        
        # 3. МГНОВЕННОЕ УМНОЖЕНИЕ МАТРИЦ (Главный буст скорости!)
        Scores = X_batch.dot(self.S_matrix)
        
        # 4. Извлекаем Топ-N кандидатов
        final_uids, final_items, final_scores = [], [], []
        
        for u_idx in range(Scores.shape[0]):
            start, end = Scores.indptr[u_idx], Scores.indptr[u_idx+1]
            if start == end: continue
            
            data = Scores.data[start:end]
            indices = Scores.indices[start:end]
            
            if len(data) > num_candidates:
                # Берем Топ-N 
                top_idx = np.argpartition(-data, num_candidates - 1)[:num_candidates]
                # Сортируем внутри Топ-N
                sort_order = np.argsort(-data[top_idx])
                final_data = data[top_idx][sort_order]
                final_indices = indices[top_idx][sort_order]
            else:
                sort_order = np.argsort(-data)
                final_data = data[sort_order]
                final_indices = indices[sort_order]
                
            final_uids.extend([batch_uids[u_idx]] * len(final_data))
            final_items.extend(self.item_id_map_raw[final_indices])
            final_scores.extend(final_data)

        return pl.DataFrame({
            "uid": np.array(final_uids, dtype=np.uint32),
            "item_id": np.array(final_items, dtype=np.uint32),
            "score": np.array(final_scores, dtype=np.float32)
        }).with_columns(pl.lit(self.name).alias("source"))

    def _empty_df(self):
        return pl.DataFrame(schema={"uid": pl.UInt32, "item_id": pl.UInt32, "score": pl.Float32, "source": pl.Utf8})


class EASEGenerator:
    # Ставим 60к, если 300ГБ памяти! Матрица 60k*60k займет ~14.4 ГБ RAM
    def __init__(self, top_k_items=60000, l2_reg=500.0, file_prefix=None):
        self.top_k_items = top_k_items
        self.l2_reg = l2_reg
        
        pr = file_prefix if file_prefix else ''
        self.base_path = Path(f"fittingdata/cg/{pr + '_' + self.name}")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.B_matrix = None
        self.item_id_map_raw = None
        self.item_to_idx_dict = None

    @property
    def name(self) -> str:
        return "ease"
    
    def _is_fitted(self) -> bool:
        return (self.base_path / "ease_weights.npy").exists() and (self.base_path / "item_map.parquet").exists()

    def fit(self, train_data: pl.LazyFrame, force: bool = False):
        if not force and self._is_fitted():
            print(f"[{self.name}] УЖЕ ОБУЧЕН! Пропускаю fit().")
            return
            
        print(f"[{self.name}] Fitting...")
        df = train_data.select(["uid", "item_id"]).collect()
        top_items_df = df.group_by("item_id").len().sort("len", descending=True).head(self.top_k_items).select("item_id")
        top_items = top_items_df["item_id"].to_numpy()
        pl.DataFrame({"item_id": top_items}).write_parquet(self.base_path / "item_map.parquet")
        
        df = df.join(top_items_df, on="item_id", how="inner")
        users = df["uid"].unique().to_numpy()
        u_map = pl.DataFrame({"uid": users}).with_row_index("u_idx").with_columns(pl.col("u_idx").cast(pl.Int32))
        i_map = pl.DataFrame({"item_id": top_items}).with_row_index("i_idx").with_columns(pl.col("i_idx").cast(pl.Int32))
        
        df = df.join(u_map, on="uid").join(i_map, on="item_id")
        X = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df["u_idx"].to_numpy(), df["i_idx"].to_numpy())), shape=(len(users), len(top_items)))
        
        del df; gc.collect()
        G = X.T.dot(X).toarray().astype(np.float32)
        del X; gc.collect()

        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.l2_reg
        
        # Ускоренная инверсия (Cholesky-based)
        print(f"[{self.name}] Inverting {self.top_k_items}x{self.top_k_items} matrix...")
        P = scipy.linalg.inv(G, overwrite_a=True, check_finite=False)
        
        B = P / (-np.diag(P))
        B[diag_indices] = 0.0
        np.save(self.base_path / "ease_weights.npy", B.astype(np.float32))

    def startup(self):
        print(f"[{self.name}] Loading weights to RAM...")
        self.B_matrix = np.load(self.base_path / "ease_weights.npy")
        self.item_id_map_raw = pl.read_parquet(self.base_path / "item_map.parquet")["item_id"].to_numpy()
        # Маппинг для векторизованной генерации
        self.item_mapping_df = pl.DataFrame({
            "item_id": self.item_id_map_raw,
            "i_idx": np.arange(len(self.item_id_map_raw), dtype=np.int32)
        })

    def generate(self, context: 'BatchContext', num_candidates: int) -> pl.DataFrame:
        u_map_df = context.target_users.select("uid").collect().with_row_index("u_batch_idx")
        batch_uids = u_map_df.get_column("uid").to_numpy().astype(np.uint32)
        
        history = (
            context.history_likes
            .join(u_map_df.lazy(), on="uid", how="inner")
            .join(self.item_mapping_df.lazy(), on="item_id", how="inner")
            .select(["u_batch_idx", "i_idx"])
            .collect()
        )
        if history.height == 0: return self._empty_df()

        n_users, n_items = len(batch_uids), len(self.item_id_map_raw)
        X_batch = sp.csr_matrix((np.ones(history.height, dtype=np.float32), (history["u_batch_idx"].to_numpy(), history["i_idx"].to_numpy())), shape=(n_users, n_items))

        chunk_size = 10000
        top_n = min(num_candidates, n_items)
        all_items, all_scores, all_uids = [], [], []

        for i in range(0, n_users, chunk_size):
            X_chunk = X_batch[i : i + chunk_size]
            if X_chunk.nnz == 0: continue
            scores_chunk = X_chunk.dot(self.B_matrix)
            
            idx_partition = np.argpartition(-scores_chunk, top_n - 1, axis=1)[:, :top_n]
            scores_partition = np.take_along_axis(scores_chunk, idx_partition, axis=1)
            idx_sort = np.argsort(-scores_partition, axis=1)
            
            final_idx = np.take_along_axis(idx_partition, idx_sort, axis=1)
            final_scores = np.take_along_axis(scores_partition, idx_sort, axis=1)
            
            actual_chunk_size = scores_chunk.shape[0]
            all_uids.append(np.repeat(batch_uids[i : i + actual_chunk_size], top_n))
            all_items.append(self.item_id_map_raw[final_idx.flatten()])
            all_scores.append(final_scores.flatten())

        if not all_uids: return self._empty_df()

        return pl.DataFrame({
            "uid": np.concatenate(all_uids).astype(np.uint32),
            "item_id": np.concatenate(all_items).astype(np.uint32),
            "score": np.concatenate(all_scores).astype(np.float32),
        }).with_columns(pl.lit(self.name).alias("source"))


    def shutdown(self):
        self.B_matrix = None
        self.item_id_map_raw = None
        self.item_to_idx_dict = None
        gc.collect()

