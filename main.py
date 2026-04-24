import gc
import polars as pl
from typing import List, Tuple

from cg.candgen import RetrievalStage
from features.feature_manager import FeatureManager
from interfaces import BatchContext


class RecSysPipeline:
    def __init__(self, retrieval_stage: RetrievalStage, feature_manager: FeatureManager):
        self.retrieval = retrieval_stage
        self.features = feature_manager

    def fit_all(self, train_history_data: pl.LazyFrame):
        """
        Обучает все генераторы и источники фичей, дампит всё на диск.
        """
        print("=== Fitting Retrieval Stage ===")
        for gen in self.retrieval.generators:
            gen.fit(train_history_data)

        print("=== Fitting Feature Manager ===")
        self.features.fit_all(train_history_data)
        print("=== Fit Complete ===")

    def create_dataset(
            self,
            target_uids: List[int],
            history_data: pl.LazyFrame,
            labels_data: pl.LazyFrame = None,
            num_candidates: int = 200,
            batch_size: int = 10000
    ) -> pl.DataFrame:
        """
        Главный метод для получения обучающей или валидационной выборки.
        Работает батчами, чтобы не съесть всю RAM.

        labels_data: Если передано, добавится колонка 'target' (1 или 0).
        """
        print(f"Starting dataset generation for {len(target_uids)} users...")

        # 1. Загружаем веса/индексы с диска в RAM/GPU
        for gen in self.retrieval.generators:
            gen.startup()
        self.features.startup()

        all_batches = []

        try:
            # 2. Идем по пользователям батчами
            for i in range(0, len(target_uids), batch_size):
                batch_uids = target_uids[i: i + batch_size]
                print(f"  Processing batch {i} to {i + batch_size}...")

                # Формируем контекст для батча
                batch_users_lf = pl.DataFrame({"uid": batch_uids}).cast({"uid": pl.UInt32}).lazy()
                context = BatchContext(
                    target_users=batch_users_lf,
                    history_likes=history_data.with_columns([
                        pl.col("uid").cast(pl.UInt32),
                        pl.col("item_id").cast(pl.UInt32)
                    ]),
                    history_listens=pl.LazyFrame()  # Заглушка на будущее
                )

                # Этап 1: Retrieval (Отбор кандидатов)
                print('    Retrieval (Отбор кандидатов)')
                candidates_lf = self.retrieval.fetch_all(context, num_candidates_per_gen=num_candidates)

                # Этап 2: Feature Engineering (Навешивание фичей)
                print('    Feature Engineering (Навешивание фичей)')
                features_lf = self.features.extract(candidates_lf, context)

                # Этап 3: Создание таргета (если это Train)
                if labels_data is not None:
                    target_lf = (
                        labels_data.select(["uid", "item_id"])
                        .unique()  # Защита от дублей в датасете лайков
                        .with_columns(pl.lit(1).cast(pl.Int8).alias("target"))
                    )
                    features_lf = (
                        features_lf
                        .join(target_lf, on=["uid", "item_id"], how="left")
                        # ИСПРАВЛЕНИЕ: заполняем нулями ТОЛЬКО колонку target!
                        .with_columns(pl.col("target").fill_null(0))
                    )

                # "Приземляем" батч в физическую память
                batch_df = features_lf.collect()
                all_batches.append(batch_df)

                # Чистим память после каждого батча
                del batch_df
                gc.collect()

        finally:
            # 3. Выгружаем модели и индексы из памяти в любом случае
            for gen in self.retrieval.generators:
                gen.shutdown()
            self.features.shutdown()
            gc.collect()

        print("Dataset generation complete!")
        # Собираем все готовые батчи в одну таблицу (она уже сжатая)
        return pl.concat(all_batches)



def calc_stats(train_dataset, raw_interactions):
    """
    train_dataset: твой сгенерированный датасет (кандидаты + таргет)
    raw_interactions: исходный data_train или data_val (все лайки периода)
    """
    gen_cols = [col for col in train_dataset.columns if col.startswith("is_")]
    total_rows = train_dataset.height
    
    # 1. Считаем "Истинный потолок" (сколько всего лайков было у этих юзеров)
    # Важно: берем лайки только тех юзеров, которые есть в нашем train_dataset
    target_users = train_dataset["uid"].unique()
    true_likes_count = (
        raw_interactions
        .filter(pl.col("uid").is_in(target_users))
        .height
    )

    print(f"=== СТАТИСТИКА ПОКРЫТИЯ (RECALL) ===")
    print(f"Всего реальных лайков в периоде: {true_likes_count}")
    
    # 2. ЭФФЕКТИВНОСТЬ (Честный Recall)
    # Сколько % от ВСЕХ лайков нашел каждый генератор
    pos_df = train_dataset.filter(pl.col("target") == 1)
    found_total = pos_df.height
    
    print(f"\nОбщий Recall системы (все генераторы вместе): {found_total/true_likes_count:.2%}")
    print(f"Потеряно лайков (не попали в кандидаты): {true_likes_count - found_total}")

    for col in gen_cols:
        hits = pos_df[col].sum()
        print(f"  {col} Recall: {hits/true_likes_count:.2%}")

    print("\n=== 3. КОНВЕРСИЯ ВНУТРИ КАНДИДАТОВ (PRECISION) ===")
    # Группируем по комбинациям флагов
    intersections = (
        train_dataset
        .group_by(gen_cols)
        .agg([
            pl.len().alias("total_in_group"),
            pl.col("target").sum().alias("likes_in_group")
        ])
        .with_columns([
            # Вероятность лайка, если трек попал в эту группу
            (pl.col("likes_in_group") / pl.col("total_in_group") * 100).round(4).alias("%_конверсия"),
            # Какой % от ВСЕХ реальных лайков закрывает эта группа
            (pl.col("likes_in_group") / true_likes_count * 100).round(2).alias("%_contribution_to_recall")
        ])
        .sort("total_in_group", descending=True)
    )

    print(intersections.sort("%_конверсия", descending=True))

# Запуск:
# calc_stats(train_dataset, data_train)
