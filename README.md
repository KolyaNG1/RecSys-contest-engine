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

Что делает:

- строит user-item матрицу взаимодействий;
- обучает `implicit.als.AlternatingLeastSquares`;
- сохраняет user/item факторы и маппинги;
- строит Faiss-индекс по item-векторам;
- для батча пользователей делает ANN-поиск top-N айтемов по inner product.

Сильная сторона:

- персонализированный retrieval с хорошим базовым покрытием.

### 2) `IALSItemToItemGenerator` (`name = "ials_i2i"`)

Что делает:

- берет последние лайки пользователя (`n_last_items`);
- для каждого лайкнутого трека ищет похожие в Faiss по iALS item-факторам;
- объединяет результаты, дедуплицирует `(uid, item_id)`, оставляет top-N на пользователя.

Сильная сторона:

- добавляет "локальную" похожесть относительно недавней истории пользователя.

### 3) `GlobalPopularityGenerator` (`name = "global_pop"`)

Что делает:

- считает глобальный топ треков по частоте лайков;
- на инференсе кросс-джойнит top-список со всеми целевыми пользователями.

Сильная сторона:

- стабильный бэкап-кандидатинг и хороший холодный baseline.

### 4) `ArtistPopularityGenerator` (`name = "artist_pop"`)

Что делает:

- по train-истории строит топ треков каждого артиста;
- по истории батча определяет любимых артистов пользователя (`top_n_artists`);
- рекомендует популярные треки этих артистов (`tracks_per_artist`).

Сильная сторона:

- персонализация через вкусы по артистам, особенно полезно при короткой истории.

### 5) `CoVisitationGenerator` (`name = "covisitation"`)

Что делает:

- строит разреженную матрицу user-item;
- считает item-item граф `X^T X`;
- обрезает до `top_k_similar` соседей на трек и сохраняет sparse similarity matrix;
- на батче умножает историю пользователя на item-item матрицу и получает score кандидатов.

Сильная сторона:

- быстро поднимает кандидатов по совместной встречаемости без latent-факторов.

### 6) `EASEGenerator` (`name = "ease"`)

Что делает:

- ограничивает пространство топ-популярными айтемами (`top_k_items`);
- строит Gram-матрицу и рассчитывает веса EASE (линейная модель item-item);
- на инференсе умножает батч-историю на матрицу весов, получает релевантности и top-N.

Сильная сторона:

- мощный item-item сигнал с хорошим качеством на implicit feedback.
