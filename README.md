# Разработка высокоэффективного майнера Pearl (PRL) — `ApexPRL`

Рабочая директория научно-исследовательской разработки и полного исходного кода собственного специализированного GPU-майнера криптовалюты **Pearl (`PRL`, алгоритм `PearlHash` / `cuPOW` / `NoisyGEMM`)** под видеокарты **NVIDIA GeForce RTX 40-й серии (`sm_89` Ada Lovelace)** и **RTX 50-й серии (`sm_120a` Blackwell)** с приоритетом под **HiveOS Linux** и пул **Kryptex (`Stratum v2 Gzip`)**.

- **Версия:** `v0.5.0` (**Persistent-`BASA` Unpredicated `Basis-128x256` + `K=65536` + SM-Aligned Layered Merkle Cache (`0.04 мс`/шара) + Single-Pass O(1) 2-Opt (`Ovh=0.32%`)** — `8` встроенных `cubin`, `0` спиллов, `0` `BRX`)
- **Комиссия разработчика (Dev Fee):** `0.00%` (полностью отсутствует).
- **Целевая ОС:** **HiveOS Linux** / любой дистрибутив Linux `x86_64` с `GLIBC >= 2.30` (совместимо с Ubuntu 20.04 / 22.04 / 24.04) и драйвером NVIDIA `>= 570.x` (для RTX 50) или `>= 535.x` (для RTX 40).
- **Приоритетный пул:** **Kryptex Pool** (`stratum+ssl://prl-ru.kryptex.network:8048`, резерв `stratum+ssl://prl-eu.kryptex.network:8048`) — нативный TLS (`rustls`), сжатие доказательств `gzip` (>20–58× экономии трафика на матрицах `Constant-64` / `Constant-0`), расчёт `compact_floor` и авто-подбор оптимальной статической сложности `d=...`.

---

## 1. Готовый релизный пакет для HiveOS Linux (`dist/`)

| Файл | Назначение | SHA-256 |
|---|---|---|
| [`dist/apexprl-hiveos.tar.gz`](dist/apexprl-hiveos.tar.gz) | Готовый архив **Custom Miner** для установки на риги HiveOS (`v0.5.0`) | `cb126f9e178c8ffe539417d3841da34aaffdb5587a779b28f8546d1a9dc458d0` |
| [`dist/apexprl`](dist/apexprl) | Автономный переносимый Linux ELF-бинарник (`GLIBC_2.30`, 8 встроенных `cubin`: `Basis128` + `Direct-L2` + `EDCP-120` + `Basis160` под `sm_89` и `sm_120a`) | `615a9fc0d548d3bd658bcd681e5e3f1f290da82314025c31707960b1edd5bd5c` |
| [`dist/flight-sheet-kryptex.json`](dist/flight-sheet-kryptex.json) | Готовый шаблон полётного листа (Flight Sheet) HiveOS под пул Kryptex (`v0.5.0`) | — |
| [`dist/SHA256SUMS.txt`](dist/SHA256SUMS.txt) | Контрольные суммы релизных артефактов | — |

### Установка в HiveOS (Custom Miner + Kryptex)

1. **Прямая ссылка для поля Installation URL в HiveOS (`v0.5.0`):**
   ```text
   https://github.com/lunin02118-ops/apexprl/releases/download/v0.5.0/apexprl-hiveos.tar.gz
   ```
   *(Либо ручная установка по SSH на риге: `cd /tmp && wget https://github.com/lunin02118-ops/apexprl/releases/download/v0.5.0/apexprl-hiveos.tar.gz && tar -xzf apexprl-hiveos.tar.gz -C /hive/miners/custom/`)*
2. В веб-панели HiveOS создайте **Flight Sheet (Полётный лист)**:
   - **Coin:** `PRL`
   - **Wallet:** ваш адрес `prl1...` или идентификатор Kryptex (`krxXXXXXXX`)
   - **Pool:** `Configure in miner`
   - **Miner:** `Custom` -> `Setup Miner Config`:
     - **Miner name:** `apexprl`
     - **Installation URL:** `https://github.com/lunin02118-ops/apexprl/releases/download/v0.5.0/apexprl-hiveos.tar.gz`
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

## 2. Что нового в `ApexPRL v0.5.0` — Движок `Persistent-BASA Unpredicated Basis-128x256`

По итогам кремниевых дуэлей (`sm_89` / `sm_120a`) и очных симметричных Stratum v2 + Pearl V3 тестов против `OneZeroMiner v1.8.6` в **`ApexPRL v0.5.0`** реализован комплекс архитектурных улучшений:

1. **Персистентный `BASA` в SMEM + 100% беспредикатный внутренний `K`-цикл ([`basis128_core.cuh`](kernels/common/basis128_core.cuh)):**
   - Инвертированный обход тайлов (`внешний tm / внутренний tn`) загружает базис `BASA` один раз на SM и удерживает его в `90,240 Б` SMEM на все `128` столбцовых тайлов `tn` (**минус `99.2%` перезагрузок `BASA`** и минимум `36+ КиБ` свободного L1-кэша на SM).
   - 4-шаговый внутренний `K`-цикл (`step<PAR, FZ>(0..3)`) полностью избавлен от предикации (`215–220` рег. на `sm_89`, `224` рег. на `sm_120a`, **`STL/LDL = 0`, `BRX = 0`, `PRMT = 0` на `sm_120a`**).
2. **`K = 65536` (`512` чанков) + Single-Pass O(1) 2-Opt (`Ovh = 0.32%`):**
   - На `Stratum v2 (gzip)` глубина редукции по умолчанию увеличена до `K = 65536` (с автоматическим откатом до `K = 4096` на plain `Stratum v1`), а однопроходная O(1) 2-opt перестановка в [`prep_kernels.cuh`](kernels/common/prep_kernels.cuh) снижает межраундовые накладные расходы до **`0.32%`**.
3. **`SM_COUNT`-выровненная сетка `M` (`0%` отброшенных строк) + мгновенный слоёный кэш Меркла (`0.04 мс`/шара):**
   - `KernelProfile::basis_128x256_sm_aligned` выбирает `M = 2 × SM_COUNT × 128 + 16`: **`M = 6160` (`sampled 6144`)** на RTX 4060 (`24 SM`, L2-базис `4.75 МиБ`) и **`M = 17936` (`sampled 17920`)** на RTX 5070 Ti (`70 SM`, L2-базис `6.20 МиБ`) — **`100.000%` заполнение волн SM** без отбрасывания краевых варпов.
   - Слоевой `Constant64MerkleCache` в [`apexprl-pow`](crates/apexprl-pow/src/lib.rs) с точной поддержкой нечётных узлов `pearl-blake3` обновляет `root_a` за **`~2.5 мкс`** и собирает полное доказательство за **`~0.04 мс`** на CPU (`0` чтений VRAM), исключая `~115 мс` простой GPU на каждую найденную шару.
4. **Очные симметричные тесты Stratum v2 + `zk-pow` V3 (`RTX 4060 Laptop @ 30.0 Вт`, протокол `20s pre-soak + ABBA`):**
   - **Автономный бенчмарк (`--benchmark 12`):** **`38.01 TH/s`** (`Kernel = 38.13 TH/s`, `Ovh = 0.32–0.33%`, `SOL = 91.6–92.5%`, **`1.269 TH/s/W`**).
   - **Дуэль ABBA при `d = 1,048,576` (`0x1b0fffff`, `80.5 с` на майнер, 100% проверка шар консенсусом Pearl V3):**
     - **`ApexPRL v0.5.0` (`0%` DevFee):** **`187 / 187` валидных V3-шар** (`139.41` чистых шар/мин, `EmpNet = 40.88 TH/s`, `TrueNet = 37.84 TH/s`, `TrueSOL = 91.84%`, `29.94 Вт`, победа в обоих заездах `98` и `89` против `80` и `82`).
     - **`OneZeroMiner v1.8.6` (`1%` DevFee):** **`162 / 162` валидных V3-шар** (`119.62` чистых шар/мин, `EmpNet = 35.07 TH/s`, `30.05 Вт`) -> **`ApexPRL v0.5.0` принёс на `+15.4%` больше валидных шар (`+16.5%` чистыми на пуле)** за счёт отсутствия `~115 мс` CPU-пауз пересборки дерева Меркла, `0%` отброшенных строк и `0%` DevFee.
   - **Дуэль ABBA при `d = 2,097,152` (`0x1b07ffff`, `80.5 с`):** **`85 / 85`** шар у `ApexPRL v0.5.0` против **`83 / 83`** у `OneZeroMiner v1.8.6` (**`+3.2%` чистых шар**).
   - **Стресс-тест высокой частоты шар при `d = 131,072` (`0x1b7fffff`, `30 с`):** **`487 / 487`** шар (`35.57 TH/s`) у `ApexPRL` против **`98 / 98`** (`7.17 TH/s`) у `OneZeroMiner` (**`4.96×` преимущество**).

---

## 3. Прямой запуск из командной строки Linux

```bash
# 1. Список обнаруженных видеокарт NVIDIA и их параметров SM / L2
./dist/apexprl --list-devices

# 2. Пятипрофильный аппаратный самотест (Basis-128 + Basis-128-SM + EDCP-120 + Direct-L2 + Basis-160: GPU == CPU == zk-pow V3 + gzip)
./dist/apexprl --self-test

# 3. Автономные бенчмарки движков v0.5.0 с автотюнингом и расчётом % от SOL
./dist/apexprl --benchmark 15 --tune --engine basis128
./dist/apexprl --benchmark 15 --tune --engine direct
./dist/apexprl --benchmark 15 --tune --engine edcp

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

- [`Cargo.toml`](Cargo.toml) — корень Rust-workspace (`v0.5.0`):
  - [`crates/apexprl-pow/`](crates/apexprl-pow/src/lib.rs) — математика консенсуса `PearlHash V3` (Salted Seed), слоевой кэш дерева Меркла `Constant64MerkleCache` с поддержкой произвольного `SM_COUNT`-выровненного `M` (`6160×32768×65536` / `17936×32768×65536`), сборщик `PlainProof` (`~40 мкс`), встроенная пред-отправочная верификация через официальный `zk-pow` и `gzip`-сжатие.
  - [`crates/apexprl-gpu/`](crates/apexprl-gpu/src/lib.rs) — динамические `dlopen`-обёртки над `libcuda.so.1` и `libnvidia-ml.so.1`, движки `Basis128` (смежный L2-pinned буфер `d_bas_ab` `4.75–8.00 МиБ`), `EDCP-120`, `Basis160` и `DirectL2`, `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 100`, автотюнер (`--tune`) и бережное NVML-управление с сохранением внешнего OC (PenguinBurner).
  - [`crates/apexprl-stratum/`](crates/apexprl-stratum/src/lib.rs) — сетевой стек `rustls` TLS / TCP для `Kryptex Stratum v2 (gzip)` и классического `Stratum v1` с автоматическим Failover.
  - [`crates/apexprl-cli/`](crates/apexprl-cli/src/main.rs) — мульти-GPU рантайм, пул потоков сборки доказательств, бенчмарк с живым контролем `avgSM` / `% SOL` и HTTP-сервер телеметрии [`api_server.rs`](crates/apexprl-cli/src/api_server.rs).
- [`kernels/`](kernels/) — специализированные CUDA C++20 / PTX вычислительные ядра:
  - [`kernels/common/basis128_core.cuh`](kernels/common/basis128_core.cuh) — общее параметризованное ядро **v0.5.0 Persistent-A Unpredicated Basis-128x256** (`FA=8, FB=16, FR=24`, персистентный `BASA`, 100% беспредикатный 4-шаговый `K`-цикл, `0` спиллов, `0` `BRX`).
  - [`kernels/apexprl_b128_sm120a.cu`](kernels/apexprl_b128_sm120a.cu) & [`kernels/apexprl_b128_sm89.cu`](kernels/apexprl_b128_sm89.cu) — варианты под RTX 50 (`sm_120a`, `224` рег., `elect.sync` TMA `UBLKCP`) и RTX 40 (`sm_89`, `215–220` рег., `cp.async.cg`).
  - [`kernels/apexprl_edcp_sm120a.cu`](kernels/apexprl_edcp_sm120a.cu) & [`kernels/apexprl_edcp_sm89.cu`](kernels/apexprl_edcp_sm89.cu) — движок `EDCP-120` (`--engine edcp`).
  - [`kernels/apexprl_direct_sm120a.cu`](kernels/apexprl_direct_sm120a.cu) & [`kernels/apexprl_direct_sm89.cu`](kernels/apexprl_direct_sm89.cu) — движок `Direct-L2 128×128` (`--engine direct`).
  - [`kernels/apexprl_sm120a.cu`](kernels/apexprl_sm120a.cu) & [`kernels/apexprl_sm89.cu`](kernels/apexprl_sm89.cu) — резервные ядра `Basis-160x256` (`--engine basis`).
- [`hiveos/`](hiveos/) — скрипты интеграции с HiveOS (`h-manifest.conf`, `h-config.sh`, `h-run.sh`, `h-stats.sh`, `flight-sheet-kryptex.json`).
- [`scripts/package_hiveos.sh`](scripts/package_hiveos.sh) и [`tools/check_sass.py`](tools/check_sass.py) — автоматизированный SASS-аудит всех 8 `cubin` (строго `0` спиллов и `0` `BRX`), сборка с переносимостью `GLIBC_2.30` и упаковка `dist/apexprl-hiveos.tar.gz`.

---

## 5. Навигация по аналитической и архитектурной документации

1. **[Сводный итоговый отчёт НИОКР](ОТЧЕТ_ИССЛЕДОВАНИЕ_И_АРХИТЕКТУРА_МАЙНЕРА_PEARL.md)** — главный обзорный документ с таблицами сравнения майнеров и расчётами энергоэффективности.
2. **[01. Исследование экосистемы и алгоритма Pearl (`cuPOW`)](docs/01_Исследование_экосистемы_и_алгоритма_Pearl.md)** — математическая модель `NoisyGEMM`, `cert_version = 3` (Salted Seed), штраф за ранг (`r = 128`) и подготовка к `cert_version = 4` (FP8).
3. **[02. Глубокий анализ существующих майнеров](docs/02_Анализ_существующих_майнеров.md)** — сравнительный аудит коммерческих и открытых майнеров.
4. **[03. Микроархитектура NVIDIA RTX 40 (`sm_89`) и RTX 50 (`sm_120`)](docs/03_Микроархитектура_RTX40_Ada_и_RTX50_Blackwell.md)** — физические пределы `IMMA.16832` (1024 MAC/такт/SM), иерархия L2-кэша и правила компиляции `ptxas`.
5. **[04. Концепция и архитектура собственного майнера `ApexPRL`](docs/04_Концепция_и_архитектура_собственного_майнера.md)** — детальная архитектура модулей и вычислительного конвейера.
6. **[05. Спецификация HiveOS Linux и приоритетного пула Kryptex (`Stratum v2 Gzip`)](docs/05_Спецификация_HiveOS_Linux_и_Kryptex_Stratum_v2.md)** — побитовый протокол Kryptex и спецификация переносимости под образы HiveOS.
