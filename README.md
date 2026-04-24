# RecSys-contest-engine

Контестное решение задачи рекомендаций для Yandex Music (Yambda): предсказание треков, которые пользователь с высокой вероятностью лайкнет в будущем.

Проект реализован как двухэтапный рекомендательный пайплайн:

1. **Candidate Generation (retrieval)** - быстрый отбор релевантных кандидатов из большого каталога.
2. **Ranking** - обучение модели ранжирования (CatBoost), которая сортирует кандидатов и формирует top-100.

---

## Задача

Для каждого `uid` нужно выдать список из 100 `item_id`, отсортированных по вероятности лайка.  
Метрика соревнования: **Recall@100**.

Формат сабмита:

```csv
uid,item_ids
1,10 20 30 ...
2,8 44 11 ...
```

---

## Архитектура решения

### 1) Candidate Generation

Реализовано в `cg/candgen.py`. Используются несколько генераторов кандидатов:

- `IALSCandidateGenerator` - user-item retrieval на базе iALS (implicit ALS + Faiss).
- `IALSItemToItemGenerator` - item-to-item retrieval от последних лайков пользователя.
- `GlobalPopularityGenerator` - глобально популярные треки.
- `ArtistPopularityGenerator` - топ треков любимых артистов пользователя.
- `CoVisitationGenerator` - item-item похожесть через co-visitation (`X^T X`).
- `EASEGenerator` - линейная item-item модель EASE.

`RetrievalStage` объединяет кандидатов из всех источников, дедуплицирует пары `(uid, item_id)`, агрегирует score и добавляет флаги источников (`is_ials`, `is_global_pop`, ...). Затем фильтрует уже лайкнутые айтемы через anti-join с историей.

### 2) Feature Engineering

Реализовано в `features/feature_manager.py` через `FeatureManager` и набор `FeatureSource`.

Ключевые источники фичей:

- `ItemStaticFeatureSource` - статические признаки трека (`item_cnt_likes`, `item_organic_ratio`, artist metadata).
- `UserStaticFeatureSource` - статические признаки пользователя (`user_total_likes`, `user_unique_items`).
- `IALSDotProductSource` - дополнительный сигнал от ALS (`als_dot`, `als_rank`).
- `ItemTrendFeatureSource` - трендовость трека по свежим лайкам (`item_trend_ratio`).
- `ArtistStaticFeatureSource` - агрегаты по артисту.
- `UserArtistAffinityFeatureSource` - аффинити пользователя к артисту (`user_artist_likes_cnt`).
- `CandidateSourceFeatureSource` - количество источников кандидата (`source_count`).

### 3) Ranking

Реализован в `pipeline.ipynb`:

- формируется train/val candidate dataset через `RecSysPipeline.create_dataset(...)`;
- добавляется таргет (`target=1` для пар из label-окна, иначе `0`);
- обучается `CatBoostClassifier`;
- предсказания сортируются и режутся до top-100 на пользователя.

---

## Структура репозитория

```text
.
├─ main.py                     # Core pipeline orchestration (fit + build dataset)
├─ interfaces.py               # BatchContext и интерфейсы данных батча
├─ cg/
│  ├─ __init__.py
│  └─ candgen.py               # Все retrieval генераторы
├─ features/
│  ├─ __init__.py
│  └─ feature_manager.py       # FeatureManager и feature sources
├─ pipeline.ipynb              # Основные эксперименты, train/val/infer/submission
├─ iALS_fitting.ipynb          # Предобучение/прогрев iALS артефактов
└─ fittingdata/
   ├─ cg/                      # Кэш retrieval артефактов (factors, indices, maps)
   └─ features/                # Кэш feature артефактов (parquet-статистики)
```

---

## Как работает пайплайн

### `main.py`

`RecSysPipeline`:

- `fit_all(train_history_data)`  
  Обучает/предрасчитывает retrieval-генераторы и feature-источники, сохраняет артефакты в `fittingdata/...`.

- `create_dataset(target_uids, history_data, labels_data=None, num_candidates=200, batch_size=10000)`  
  Батчево:
  1) запускает retrieval;  
  2) обогащает кандидатов фичами;  
  3) при наличии `labels_data` добавляет бинарный `target`;  
  4) собирает итоговый DataFrame.

Также есть утилита `calc_stats(...)` для анализа покрытия кандидатов и вкладов источников.

### `interfaces.py`

`BatchContext` - единый объект контекста батча:

- `target_users`
- `history_likes`
- `history_listens` (заглушка под будущее расширение)

---

## Данные и артефакты

В ноутбуке используются данные Yambda/KaggleHub (лайки, mapping трек-артист, test users).  
После `fit_all(...)` на диск сохраняются:

- ALS-факторы пользователей и треков;
- Faiss-индексы;
- item/user mapping таблицы;
- top/popularity таблицы;
- feature-статистики (item/user/artist/trend).

Это ускоряет повторные прогоны экспериментов и инференс.

---

## Быстрый старт

Проект сейчас ориентирован на запуск через ноутбуки.

1. Установить зависимости:

```bash
pip install polars numpy scipy implicit faiss-cpu catboost kagglehub torch pyarrow jupyter
```

2. Запустить Jupyter:

```bash
jupyter notebook
```

3. Открыть `pipeline.ipynb` и выполнить ячейки последовательно:

- загрузка данных;
- настройка retrieval + feature sources;
- `pipeline.fit_all(...)`;
- сбор train/val через `create_dataset(...)`;
- обучение CatBoost;
- расчет Recall@100;
- генерация submission.

Опционально: предварительно прогреть iALS-кэши в `iALS_fitting.ipynb`.

---

## Технологии

- Python 3.12
- Polars (LazyFrame/DataFrame pipeline)
- NumPy / SciPy (sparse algebra)
- implicit (ALS)
- Faiss (быстрый nearest neighbors retrieval)
- CatBoost (ranking/classification stage)
- Jupyter Notebook (эксперименты и сборка сабмита)

---

## Возможности кода

### Управление пайплайном

- Единый оркестратор `RecSysPipeline` для полного цикла: `fit_all(...)` и `create_dataset(...)`.
- Батчевая обработка пользователей (`batch_size`) для контроля потребления памяти на больших выборках.
- Явные `startup()/shutdown()` для генераторов и фичей: загрузка тяжелых артефактов только на время расчета.
- Работа с `polars.LazyFrame`, что позволяет откладывать вычисления и собирать только нужные данные в конце батча.

### Подготовка train/val/test датасетов

- Построение единого candidate dataset с колонками `uid`, `item_id`, `score`, `is_*`.
- Автоматическая сборка таргета при передаче `labels_data`:
  - положительный класс (`target=1`) для пар из label-окна;
  - отрицательный класс (`target=0`) для остальных кандидатов.
- Исключение уже известных лайков через anti-join (`history_likes`) на этапе retrieval.
- Утилита `calc_stats(...)` для анализа recall-покрытия генераторов и пересечений источников.

### Кэширование артефактов

- Сохранение retrieval-артефактов в `fittingdata/cg/...`:
  - факторы ALS (`user_factors.npy`, `item_factors.npy`);
  - индексы Faiss (`index.bin` / `index_hnsw_flat.bin`);
  - маппинги id (`user_map.parquet`, `item_map.parquet`);
  - подготовленные таблицы популярности.
- Сохранение feature-артефактов в `fittingdata/features/...`:
  - `item_features.parquet`, `user_features.parquet`, `artist_features.parquet`, `item_trend.parquet`.
- Повторные прогоны экспериментов работают быстрее за счет переиспользования уже рассчитанного кэша.

---

## Candidate Generators (CG)

Все генераторы реализуют единый интерфейс (`fit`, `startup`, `generate`, `shutdown`) и возвращают кандидатов в формате:

- `uid`
- `item_id`
- `score`
- `source` (внутренний маркер источника)

После объединения в `RetrievalStage` каждый кандидат дополнительно получает бинарные флаги `is_<generator>`.

### 1) `IALSCandidateGenerator` (`name = "ials"`)

Назначение: user-to-item retrieval на latent-факторах.

Параметры конструктора:

- `factors`, `iterations`, `alpha`, `regularization`, `file_prefix`.

`fit(train_data)`:

- строит user-item CSR-матрицу из `uid`, `item_id`;
- обучает `implicit.als.AlternatingLeastSquares`;
- сохраняет:
  - `user_map.parquet`, `item_map.parquet`;
  - `user_factors.npy`, `item_factors.npy`;
  - Faiss-индекс `index.bin` по item-факторам.

`startup()`:

- загружает маппинги и факторы в память;
- поднимает Faiss-индекс для ANN-поиска.

`generate(context, num_candidates)`:

- матчится список `target_users` с известными `uid` из `user_map`;
- делает батчевый nearest-neighbors поиск top-K по user-векторам;
- возвращает `uid`, `item_id`, `score`, `source`.

### 2) `IALSItemToItemGenerator` (`name = "ials_i2i"`)

Назначение: item-to-item retrieval от последних взаимодействий пользователя.

Параметры конструктора:

- `n_last_items`, `n_similar_per_item`, `file_prefix`.

Зависимости:

- использует артефакты базового iALS (`item_factors.npy`, `item_map.parquet`, Faiss-индекс).

`startup()`:

- загружает item-факторы и индекс;
- строит словарь `item_id -> factor_index`.

`generate(context, num_candidates)`:

- выбирает последние `n_last_items` из `history_likes` для каждого `uid`;
- для каждого item делает поиск похожих в индексе;
- объединяет результаты, делает `unique(uid, item_id)`, сортирует по score, режет top-N на пользователя;
- возвращает `uid`, `item_id`, `score`, `source`.

### 3) `GlobalPopularityGenerator` (`name = "global_pop"`)

Назначение: неперсонализированный retrieval по глобальной частоте лайков.

Параметры конструктора:

- `top_n`.

`fit(train_data)`:

- считает частоты `item_id`;
- сохраняет top-`top_n` в `top_items.parquet` с колонками `item_id`, `score`.

`generate(context, num_candidates)`:

- делает cross join `target_users x top_items`;
- возвращает `uid`, `item_id`, `score`, `source`.

### 4) `ArtistPopularityGenerator` (`name = "artist_pop"`)

Назначение: retrieval по предпочтительным артистам пользователя.

Параметры конструктора:

- `artist_mapping`, `top_n_artists`, `tracks_per_artist`.

`fit(train_data)`:

- сохраняет `artist_mapping.parquet`;
- считает топ треков по каждому артисту;
- сохраняет `artist_top_tracks.parquet`.

`generate(context, num_candidates)`:

- по `history_likes` батча считает top-`top_n_artists` артистов пользователя;
- джойнит с `artist_top_tracks` и получает кандидатов;
- возвращает `uid`, `item_id`, `score` (`artist_score`), `source`.

### 5) `CoVisitationGenerator` (`name = "covisitation"`)

Назначение: item-item retrieval через co-occurrence в истории взаимодействий.

Параметры конструктора:

- `top_k_similar`, `history_depth`, `file_prefix`.

`fit(train_data)`:

- строит CSR user-item матрицу;
- считает `G = X^T X`, зануляет диагональ;
- для каждой строки оставляет top-`top_k_similar` соседей;
- сохраняет:
  - `similarity_matrix.npz`;
  - `item_map.parquet`.

`generate(context, num_candidates)`:

- собирает историю батча глубиной `history_depth`;
- кодирует ее как `X_batch`;
- считает `Scores = X_batch * similarity_matrix`;
- извлекает top-N по каждому пользователю;
- возвращает `uid`, `item_id`, `score`, `source`.

### 6) `EASEGenerator` (`name = "ease"`)

Назначение: линейный item-item retrieval по матрице весов EASE.

Параметры конструктора:

- `top_k_items`, `l2_reg`, `file_prefix`.

`fit(train_data)`:

- выбирает top-`top_k_items` по популярности;
- строит CSR user-item матрицу в этом подпространстве;
- считает Gram-матрицу `G = X^T X`, добавляет `l2_reg` на диагональ;
- инвертирует `G`, строит матрицу весов `B`;
- сохраняет:
  - `ease_weights.npy`;
  - `item_map.parquet`.

`generate(context, num_candidates)`:

- кодирует историю батча в матрицу `X_batch`;
- считает релевантности `scores = X_batch * B` (чанками);
- извлекает top-N по каждому пользователю;
- возвращает `uid`, `item_id`, `score`, `source`.
