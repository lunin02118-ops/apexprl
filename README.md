# Разработка высокоэффективного майнера Pearl (PRL) — `ApexPRL`

Рабочая директория научно-исследовательской разработки и полного исходного кода собственного специализированного GPU-майнера криптовалюты **Pearl (`PRL`, алгоритм `PearlHash` / `cuPOW` / `NoisyGEMM`)** под видеокарты **NVIDIA GeForce RTX 40-й серии (`sm_89` Ada Lovelace)** и **RTX 50-й серии (`sm_120a` Blackwell)** с приоритетом под **HiveOS Linux** и пул **Kryptex (`Stratum v2 Gzip`)**.

- **Версия:** `v0.3.0` (**Uniform-3 Basis-128x256 Engine** — `0` спиллов, `8.00 МиБ` L2-Pin под `810 МГц` GDDR7)
- **Комиссия разработчика (Dev Fee):** `0.00%` (полностью отсутствует).
- **Целевая ОС:** **HiveOS Linux** / любой дистрибутив Linux `x86_64` с `GLIBC >= 2.30` (совместимо с Ubuntu 20.04 / 22.04 / 24.04) и драйвером NVIDIA `>= 570.x` (для RTX 50) или `>= 535.x` (для RTX 40).
- **Приоритетный пул:** **Kryptex Pool** (`stratum+ssl://prl-ru.kryptex.network:8048`, резерв `stratum+ssl://prl-eu.kryptex.network:8048`) — нативный TLS (`rustls`), сжатие доказательств `gzip` (>20× экономии трафика на матрицах `Constant-64`), расчёт `compact_floor` и авто-подбор оптимальной статической сложности `d=...`.

---

## 1. Готовый релизный пакет для HiveOS Linux (`dist/`)

| Файл | Назначение | SHA-256 |
|---|---|---|
| [`dist/apexprl-hiveos.tar.gz`](dist/apexprl-hiveos.tar.gz) | Готовый архив **Custom Miner** для установки на риги HiveOS (`v0.3.0`) | `c10bbeb8e9e7a0eb86bfd1f255cd26a546a9a2eae556578f7a4a293a1a95bc6b` |
| [`dist/apexprl`](dist/apexprl) | Автономный переносимый Linux ELF-бинарник (`GLIBC_2.30`, 6 встроенных `cubin`: `Basis128` + `Basis160` + `Direct-L2` под `sm_89` и `sm_120a`) | `8ffb274331eb46fece8f1feb021d38dfaf283a97a24f21ed4f9b3eb4537efd6b` |
| [`dist/flight-sheet-kryptex.json`](dist/flight-sheet-kryptex.json) | Готовый шаблон полётного листа (Flight Sheet) HiveOS под пул Kryptex (`v0.3.0`) | — |
| [`dist/SHA256SUMS.txt`](dist/SHA256SUMS.txt) | Контрольные суммы релизных артефактов | — |

### Установка в HiveOS (Custom Miner + Kryptex)

1. **Прямая ссылка для поля Installation URL в HiveOS (`v0.3.0`):**
   ```text
   https://github.com/lunin02118-ops/apexprl/releases/download/v0.3.0/apexprl-hiveos.tar.gz
   ```
   *(Либо ручная установка по SSH на риге: `cd /tmp && wget https://github.com/lunin02118-ops/apexprl/releases/download/v0.3.0/apexprl-hiveos.tar.gz && tar -xzf apexprl-hiveos.tar.gz -C /hive/miners/custom/`)*
2. В веб-панели HiveOS создайте **Flight Sheet (Полётный лист)**:
   - **Coin:** `PRL`
   - **Wallet:** ваш адрес `prl1...` или идентификатор Kryptex (`krxXXXXXXX`)
   - **Pool:** `Configure in miner`
   - **Miner:** `Custom` -> `Setup Miner Config`:
     - **Miner name:** `apexprl`
     - **Installation URL:** `https://github.com/lunin02118-ops/apexprl/releases/download/v0.3.0/apexprl-hiveos.tar.gz`
     - **Hash algorithm:** `pearlhash`
     - **Wallet and worker template:** `%WAL%.%WORKER_NAME%`
     - **Pool URL:**
       ```text
       stratum+ssl://prl-ru.kryptex.network:8048
       stratum+ssl://prl-eu.kryptex.network:8048
       ```
     - **Pass:** `auto` *(майнер сам рассчитает оптимальный `d=<N>` под число SM рига, либо укажите вручную, например `d=2097152`)*
     - **Extra config arguments:**
       ```text
       --lock-mem-idle --tune
       ```
       *(Флаг `--lock-mem-idle` автоматически распознаёт внешний лок памяти **PenguinBurner / HiveOS OC** `810 МГц`, сохраняет его без перезаписи и никогда не сбрасывает при остановке майнера.)*

---

## 2. Что нового в `ApexPRL v0.3.0` — Движок `Uniform-3 Basis-128x256`

После боевого теста `v0.2.0` на риге **2 × RTX 5070 Ti (`70 SM`, `48 МиБ L2`, `2887 / 810 МГц`)** была проведена полная диагностика обеих архитектур (`v0.2.0 Direct-L2` — `163.7 TH/s`, `39.6%` SOL; `v0.1.0 Basis-160` — `337.8 TH/s`, `81.6%` SOL):

- **Причина провала `v0.2.0 Direct-L2` на залоченной `810 МГц` GDDR7:** на `256`-битной шине при `810 МГц` пропускная способность DRAM падает до **`~51.8 ГБ/с`**. Материализация панелей (`d_panel_b = 32 МиБ` + `d_panel_a = 6.5 МиБ` + `d_bas_a = 4 МиБ` = `42.5 МиБ` внутри `48 МиБ` L2 без `cuStreamSetAttribute` L2-закрепления) вытесняла матрицу `B` из L2 потоком записей `A`, упираясь в `51.8 ГБ/с` DRAM (`~81.6 TH/s/GPU`), а отсутствие `PREFERRED_SHARED_MEMORY_CARVEOUT = 100` дополнительно ограничивало `Direct-L2` до 1 CTA/SM.
- **Причина недобора `v0.1.0 Basis-160` на Blackwell (`81.6%` SOL):** сетка `160×256` требовала `26` фрагментов на `8` варпов (`RQ = 4`, перекос загрузки **`26/32 = 81.25%`**), из-за чего `ptxas` на `sm_120a` вставлял **`148 × WARPSYNC.ALL`** (`12` внутри внутреннего цикла) вокруг предицированных `@P1 LDSM`, а `MI=5, NJ=8` (`160` регистров аккумулятора) переполнял 255 регистров и порождал **`16B` спиллы в Local Memory (`3 × LDL.LU` в `K`-цикле поверх `810 МГц` GDDR7)**.

В **`ApexPRL v0.3.0`** разработан и установлен по умолчанию новый вычислительный движок **`--engine basis128` (`Uniform-3 Basis-128x256`)**, совмещающий полную независимость от частоты GDDR7 с предельным КПД тензорных ядер Blackwell и Ada:

1. **Идеальная симметрия варпов `Uniform-3` (`128×256`, `FA=8`, `FB=16`, `FR=24 = 8 × 3`):**
   - Каждый из 8 варпов собирает ровно `3` фрагмента без единого предицированного `LDSM` (`100%` баланс вместо `81.25%`).
   - Внутренний `K`-цикл содержит **`0` инструкций `WARPSYNC.ALL`** (всего `12` на 4 ядра `sm_120a` против `148` в `v0.1.0`).
2. **Строго `0` регистровых спиллов (`STL/LDL = 0`) на `sm_120a` и `sm_89`:**
   - Сокращение тайла `MI=4, NJ=8` (`128` регистров аккумулятора вместо `160`) опустило расход регистров на `sm_120a` до **`230` рег.** (`-Xptxas -O1`) с **абсолютно нулевыми спиллами (`0` байт стека, `0` обращений `STL/LDL` к `810 МГц` GDDR7)**.
   - Автоматический перебор расписания SASS под `sm_120a` отобрал 4 лучших варианта (`bw128-c9`, `bw128-c8`, `bw128-c0`, `bw128-c6`) со **статической эффективностью внутреннего цикла до `97.71%` SOL**.
3. **Единый смежный L2-базис `8.00 МиБ` с аппаратным закреплением (`L2-Pin`):**
   - Базисы `A` и `B` размещены в одном смежном буфере **`d_bas_ab = 8.00 МиБ`** (`(32768 + 32768) × 128 Б`), который принудительно закрепляется в L2-кэше (`CU_LIMIT_PERSISTING_L2_CACHE_SIZE` + `cuStreamSetAttribute(CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, PERSISTING, hitRatio=1.0)`).
   - Никаких промежуточных панелей в памяти: весь рабочий набор занимает `16.7%` от `48 МиБ` L2 на RTX 5070 Ti и `25%` от `32 МиБ` L2 на RTX 4060, обеспечивая `0%` деградации при `810 МГц` GDDR7.
4. **Оптимизации конвейера, SMEM и NVML:**
   - 4-канальное параллельное сворачивание `LOP3.LUT` (убирает последовательную цепочку зависимостей `hacc`), одноинструкционный `elect.sync` TMA (`UBLKCP`) на `sm_120a`, редкий опрос `abort_epoch` раз в 16 волн.
   - Установлен `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 100` на всех ядрах (`Basis128`, `Basis160`, `DirectL2`).
   - В `nvml_drv.rs` полностью сохраняется внешний лок памяти `700–1050 МГц` (PenguinBurner `810 МГц`) без сброса при выходе.
5. **Верифицированные результаты (`RTX 4060 Laptop`, `sm_89`, `24 SM`, `32 МиБ L2`, лимит `30.0 Вт`):**
   - **Тройной самотест Gate G0 (`--self-test`):** 100% побитовое совпадение GPU == эталонный CPU == официальный верификатор `zk-pow` V3 + `gzip` одновременно для **`basis-128`** (`1024×1024×4096` с проверкой `stp >= 16` `rotl_xor<13>`, `451/451`, 4 варианта, `33.1×`), **`basis-160`** (`242/242`, 4 варианта, `20.4×`) и **`direct-l2`** (`105/105`, 3 варианта, `21.3×`).
   - **Сравнение на живом кремнии (`30.0 Вт`):**
     - **`v0.3.0 basis128` (`ada128-c6`, `L2pin=8.00MiB`):** **`37.20 TH/s`** (`SOL=89.5%`, `Ovh=0.03%`, **`1.241 TH/s/W`**) — **`+5.6%`** к скорости и энергоэффективности поверх `v0.1.0` даже на Ada Lovelace (`sm_89`), где не было 16B-спиллов.
     - `v0.1.0 basis160` (`ada-u6`, `L2pin=8.00MiB`): `35.22 TH/s` (`SOL=85.8%`, `Ovh=0.03%`, `1.175 TH/s/W`).
     - `v0.2.0 direct` (`hl2-2cta`, `L2pin=16.0MiB`, память `8000 МГц`): `33.09 TH/s` (`SOL=91.3%`, `1.109 TH/s/W`).

---

## 3. Прямой запуск из командной строки Linux

```bash
# 1. Список обнаруженных видеокарт NVIDIA и их параметров SM / L2
./dist/apexprl --list-devices

# 2. Тройной аппаратный самотест (Basis-128 + Basis-160 + Direct-L2: GPU == CPU == zk-pow V3 + gzip)
./dist/apexprl --self-test

# 3. Автономный бенчмарк Uniform-3 Basis-128x256 (v0.3.0, по умолчанию) с автотюнингом и расчётом % от SOL
./dist/apexprl --benchmark 15 --tune

# (Опционально: сравнить с резервными движками --engine basis или --engine direct)
./dist/apexprl --benchmark 15 --tune --engine basis
./dist/apexprl --benchmark 15 --tune --engine direct

# 4. Боевой запуск на пул Kryptex (TLS 8048 + Stratum v2 Gzip + авто-сохранение лока 810 МГц)
./dist/apexprl \
  --pool stratum+ssl://prl-ru.kryptex.network:8048 \
  --pool stratum+ssl://prl-eu.kryptex.network:8048 \
  --wallet krxXXXXXXX.rig01 \
  --pass auto \
  --lock-mem-idle \
  --tune
```

Локальная телеметрия доступна по адресам:
- `http://127.0.0.1:4079/` — встроенный HTML-дашборд рига;
- `http://127.0.0.1:4079/api/v1/hiveos` — JSON-эндпоинт агента `h-stats.sh` HiveOS;
- `http://127.0.0.1:4079/api/summary` — сводная JSON-статистика.

---

## 4. Структура исходного кода

- [`Cargo.toml`](Cargo.toml) — корень Rust-workspace (`v0.3.0`):
  - [`crates/apexprl-pow/`](crates/apexprl-pow/src/lib.rs) — математика консенсуса `PearlHash V3` (Salted Seed), единый кэш дерева Меркла для `Basis128` (`32640×32768×16384`), `Basis160` и `DirectL2`, сборщик `PlainProof`, встроенная пред-отправочная верификация через официальный `zk-pow` и `gzip`-сжатие.
  - [`crates/apexprl-gpu/`](crates/apexprl-gpu/src/lib.rs) — динамические `dlopen`-обёртки над `libcuda.so.1` и `libnvidia-ml.so.1`, движки `Basis128` (единый смежный `8.00 МиБ` L2-pinned буфер `d_bas_ab`), `Basis160` и `DirectL2`, `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 100`, автотюнер (`--tune`) и бережное NVML-управление с сохранением внешнего OC (PenguinBurner).
  - [`crates/apexprl-stratum/`](crates/apexprl-stratum/src/lib.rs) — сетевой стек `rustls` TLS / TCP для `Kryptex Stratum v2 (gzip)` и классического `Stratum v1` с автоматическим Failover.
  - [`crates/apexprl-cli/`](crates/apexprl-cli/src/main.rs) — мульти-GPU рантайм, пул потоков сборки доказательств, бенчмарк с живым контролем `avgSM` / `% SOL` и HTTP-сервер телеметрии [`api_server.rs`](crates/apexprl-cli/src/api_server.rs).
- [`kernels/`](kernels/) — специализированные CUDA C++20 / PTX вычислительные ядра:
  - [`kernels/common/basis128_core.cuh`](kernels/common/basis128_core.cuh) — общее параметризованное ядро **v0.3.0 Uniform-3 Basis-128x256** (`FA=8, FB=16, FR=24`, 4-lane `LOP3.LUT` fold, редкий опрос `abort_epoch`, 0 предикации).
  - [`kernels/apexprl_b128_sm120a.cu`](kernels/apexprl_b128_sm120a.cu) — 4 варианта под RTX 50 (`sm_120a`): `bw128-c9`, `bw128-c8`, `bw128-c0`, `bw128-c6` (`-Xptxas -O1`, `230` рег., `0` спиллов, `elect.sync` TMA `UBLKCP`, `97.71%` статический SOL).
  - [`kernels/apexprl_b128_sm89.cu`](kernels/apexprl_b128_sm89.cu) — 4 варианта под RTX 40 (`sm_89`): `ada128-c9`, `ada128-c8`, `ada128-c0`, `ada128-c6` (`-Xptxas -O1`, `cp.async.cg`, `0` спиллов).
  - [`kernels/apexprl_direct_sm120a.cu`](kernels/apexprl_direct_sm120a.cu) & [`kernels/apexprl_direct_sm89.cu`](kernels/apexprl_direct_sm89.cu) — движок `Direct-L2 128×128` (`--engine direct`).
  - [`kernels/apexprl_sm120a.cu`](kernels/apexprl_sm120a.cu) & [`kernels/apexprl_sm89.cu`](kernels/apexprl_sm89.cu) — резервные ядра `Basis-160x256` (`--engine basis`).
- [`hiveos/`](hiveos/) — скрипты интеграции с HiveOS (`h-manifest.conf`, `h-config.sh`, `h-run.sh`, `h-stats.sh`, `flight-sheet-kryptex.json`).
- [`scripts/package_hiveos.sh`](scripts/package_hiveos.sh) и [`tools/check_sass.py`](tools/check_sass.py) — автоматизированный SASS-аудит всех 6 `cubin` (строго `0` спиллов и `WARPSYNC <= 16` на `Basis128`), сборка через `cargo zigbuild --target x86_64-unknown-linux-gnu.2.31` и упаковка `dist/apexprl-hiveos.tar.gz`.

---

## 5. Навигация по аналитической и архитектурной документации

1. **[Сводный итоговый отчёт НИОКР](ОТЧЕТ_ИССЛЕДОВАНИЕ_И_АРХИТЕКТУРА_МАЙНЕРА_PEARL.md)** — главный обзорный документ с таблицами сравнения майнеров и расчётами энергоэффективности.
2. **[01. Исследование экосистемы и алгоритма Pearl (`cuPOW`)](docs/01_Исследование_экосистемы_и_алгоритма_Pearl.md)** — математическая модель `NoisyGEMM`, `cert_version = 3` (Salted Seed), штраф за ранг (`r = 128`) и подготовка к `cert_version = 4` (FP8).
3. **[02. Глубокий анализ существующих майнеров](docs/02_Анализ_существующих_майнеров.md)** — сравнительный аудит коммерческих и открытых майнеров.
4. **[03. Микроархитектура NVIDIA RTX 40 (`sm_89`) и RTX 50 (`sm_120`)](docs/03_Микроархитектура_RTX40_Ada_и_RTX50_Blackwell.md)** — физические пределы `IMMA.16832` (1024 MAC/такт/SM), иерархия L2-кэша и правила компиляции `ptxas`.
5. **[04. Концепция и архитектура собственного майнера `ApexPRL`](docs/04_Концепция_и_архитектура_собственного_майнера.md)** — детальная архитектура модулей и вычислительного конвейера.
6. **[05. Спецификация HiveOS Linux и приоритетного пула Kryptex (`Stratum v2 Gzip`)](docs/05_Спецификация_HiveOS_Linux_и_Kryptex_Stratum_v2.md)** — побитовый протокол Kryptex и спецификация переносимости под образы HiveOS.
