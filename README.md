# Разработка высокоэффективного майнера Pearl (PRL) — `ApexPRL`

Рабочая директория научно-исследовательской разработки и полного исходного кода собственного специализированного GPU-майнера криптовалюты **Pearl (`PRL`, алгоритм `PearlHash` / `cuPOW` / `NoisyGEMM`)** под видеокарты **NVIDIA GeForce RTX 40-й серии (`sm_89` Ada Lovelace)** и **RTX 50-й серии (`sm_120a` Blackwell)** с приоритетом под **HiveOS Linux** и пул **Kryptex (`Stratum v2 Gzip`)**.

- **Комиссия разработчика (Dev Fee):** `0.00%` (полностью отсутствует).
- **Целевая ОС:** **HiveOS Linux** / любой дистрибутив Linux `x86_64` с `GLIBC >= 2.30` (совместимо с Ubuntu 20.04 / 22.04 / 24.04) и драйвером NVIDIA `>= 570.x` (для RTX 50) или `>= 535.x` (для RTX 40).
- **Приоритетный пул:** **Kryptex Pool** (`stratum+ssl://prl-ru.kryptex.network:8048`, резерв `stratum+ssl://prl-eu.kryptex.network:8048`) — нативный TLS (`rustls`), сжатие доказательств `gzip` (>20× экономии трафика на матрицах `Constant-64`), расчёт `compact_floor` и авто-подбор оптимальной статической сложности `d=...`.

---

## 1. Готовый релизный пакет для HiveOS Linux (`dist/`)

| Файл | Назначение | SHA-256 |
|---|---|---|
| [`dist/apexprl-hiveos.tar.gz`](dist/apexprl-hiveos.tar.gz) | Готовый архив **Custom Miner** для установки на риги HiveOS | `fa6eb43d03f7a4a7bc72b96bfbf6db6d558802ad29d2fad0a055246775fcbc86` |
| [`dist/apexprl`](dist/apexprl) | Автономный переносимый Linux ELF-бинарник (`GLIBC_2.30`, встроенные `cubin` `sm_89` + `sm_120a`) | `6544865205dbf62b40f8d2068d0a18de2041618e8e028792810ae187e347e210` |
| [`dist/flight-sheet-kryptex.json`](dist/flight-sheet-kryptex.json) | Готовый шаблон полётного листа (Flight Sheet) HiveOS под пул Kryptex | — |
| [`dist/SHA256SUMS.txt`](dist/SHA256SUMS.txt) | Контрольные суммы релизных артефактов | — |

### Установка в HiveOS (Custom Miner + Kryptex)

1. **Прямая ссылка для поля Installation URL в HiveOS:**
   ```text
   https://github.com/lunin02118-ops/apexprl/releases/download/v0.1.0/apexprl-hiveos.tar.gz
   ```
   *(Либо ручная установка по SSH на риге: `cd /tmp && wget https://github.com/lunin02118-ops/apexprl/releases/download/v0.1.0/apexprl-hiveos.tar.gz && tar -xzf apexprl-hiveos.tar.gz -C /hive/miners/custom/`)*
2. В веб-панели HiveOS создайте **Flight Sheet (Полётный лист)**:
   - **Coin:** `PRL`
   - **Wallet:** ваш адрес `prl1...` или идентификатор Kryptex (`krxXXXXXXX`)
   - **Pool:** `Configure in miner`
   - **Miner:** `Custom` -> `Setup Miner Config`:
     - **Miner name:** `apexprl`
     - **Installation URL:** `https://github.com/lunin02118-ops/apexprl/releases/download/v0.1.0/apexprl-hiveos.tar.gz`
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

### Ожидаемые показатели на RTX 5070 (`sm_120`, 48 SM, 48 МиБ L2, GDDR7) vs `OneZeroMiner`

| Параметр | `OneZeroMiner` (v1.7.x / 1.8.0) | `ApexPRL v0.1.0` (`sm_120a` Basis L2) | Преимущество `ApexPRL` |
|---|---|---|---|
| **Вычислительное ядро** | Плотный INT8 GEMM (чтение матриц из VRAM/L2) | Он-чип сборка из базиса `8.5 МиБ` (100% в 48 МиБ L2) + `TMA` + `ldmatrix.m16n16.x2.trans.b8` | Полная разгрузка шины GDDR7 |
| **Хешрейт RTX 5070 (сток / UV ~2600 МГц)** | ~96 – 106 TH/s | **~116 – 125 TH/s** | **+16% .. +22% TH/s** |
| **Хешрейт RTX 5070 (Boost ~2900–3000 МГц)** | ~108 – 115 TH/s | **~134 – 143 TH/s** | **+20% .. +25% TH/s** |
| **Потребление памяти GDDR7** | Активна (P0/P2, ~35–45 Вт впустую) | Усыплена в P8 Idle (`--lock-mem-idle`) | **-35 .. -45 Вт на карту** |
| **Энергоэффективность (TH/s на Ватт)** | ~0.58 – 0.68 TH/s/Вт | **~1.05 – 1.25 TH/s/Вт** | **До +80% выше КПД** |
| **Комиссия разработчика (DevFee)** | 1.5% – 2.0% | **0.00%** | **+1.5% .. +2.0% чистых шар** |

---

## 2. Прямой запуск из командной строки Linux

```bash
# 1. Список обнаруженных видеокарт NVIDIA и их параметров SM / L2
./dist/apexprl --list-devices

# 2. Аппаратный самотест (GPU == эталонный CPU == официальный верификатор zk-pow V3 + gzip)
./dist/apexprl --self-test

# 3. Автономный бенчмарк полной матрицы 32768x32768x16384 с автотюнингом вариантов PTX
./dist/apexprl --benchmark 10 --tune --k 16384

# 4. Боевой запуск на пул Kryptex (TLS 8048 + Stratum v2 Gzip + блокировка памяти в P8 Idle)
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

## 3. Структура исходного кода

- [`Cargo.toml`](Cargo.toml) — корень Rust-workspace:
  - [`crates/apexprl-pow/`](crates/apexprl-pow/src/lib.rs) — математика консенсуса `PearlHash V3` (Salted Seed), кэш дерева Меркла для константной сигнальной матрицы `+64`, сборщик `PlainProof`, встроенная пред-отправочная верификация через официальный `zk-pow` и `gzip`-сжатие.
  - [`crates/apexprl-gpu/`](crates/apexprl-gpu/src/lib.rs) — динамические `dlopen`-обёртки над `libcuda.so.1` и `libnvidia-ml.so.1` (нулевая привязка к версии системного CUDA Toolkit), окно персистентности L2-кэша, zero-copy страница прерывания `abort_epoch` (`< 0.25 мс` при смене блока), автотюнер 4 вариантов раскладки инструкций и NVML-управление частотами/энергопотреблением.
  - [`crates/apexprl-stratum/`](crates/apexprl-stratum/src/lib.rs) — сетевой стек `rustls` TLS / TCP для `Kryptex Stratum v2 (gzip)` и классического `Stratum v1` с автоматическим Failover.
  - [`crates/apexprl-cli/`](crates/apexprl-cli/src/main.rs) — мульти-GPU рантайм, пул потоков сборки доказательств и HTTP-сервер телеметрии [`api_server.rs`](crates/apexprl-cli/src/api_server.rs).
- [`kernels/`](kernels/) — специализированные CUDA C++20 / PTX вычислительные ядра (`-Xptxas -O1`, 0 внутренних регистровых спиллов):
  - [`kernels/apexprl_sm120a.cu`](kernels/apexprl_sm120a.cu) — нативное ядро под **RTX 50 (`sm_120a` Blackwell)**: асинхронный `TMA` (`cp.async.bulk`) + аппаратная транспозиция `ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8` + `IMMA.16832.U8.U8`.
  - [`kernels/apexprl_sm89.cu`](kernels/apexprl_sm89.cu) — нативное ядро под **RTX 40 (`sm_89` Ada Lovelace)**: пред-интерливированный базис в 8.5 МиБ L2-кэша + `cp.async` + `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` + 4 парных `__byte_perm` (`prmt.b32`) + `IMMA.16832.U8.U8`.
  - [`kernels/common/`](kernels/common/) — регистровый `BLAKE3` (`blake3_regs.cuh`), генерация базиса и подавление конфликтов банков Shared Memory (`prep_kernels.cuh`), параметризованное ядро `160×256` (`basis_gemm_core.cuh`).
- [`hiveos/`](hiveos/) — скрипты интеграции с HiveOS (`h-manifest.conf`, `h-config.sh`, `h-run.sh`, `h-stats.sh`, `flight-sheet-kryptex.json`).
- [`scripts/package_hiveos.sh`](scripts/package_hiveos.sh) и [`tools/check_sass.py`](tools/check_sass.py) — автоматизированная проверка SASS-инструкций `cubin`, сборка через `cargo zigbuild --target x86_64-unknown-linux-gnu.2.31` и упаковка `dist/apexprl-hiveos.tar.gz`.

---

## 4. Навигация по аналитической и архитектурной документации

1. **[Сводный итоговый отчёт НИОКР](ОТЧЕТ_ИССЛЕДОВАНИЕ_И_АРХИТЕКТУРА_МАЙНЕРА_PEARL.md)** — главный обзорный документ с таблицами сравнения майнеров и расчётами энергоэффективности.
2. **[01. Исследование экосистемы и алгоритма Pearl (`cuPOW`)](docs/01_Исследование_экосистемы_и_алгоритма_Pearl.md)** — математическая модель `NoisyGEMM`, `cert_version = 3` (Salted Seed), штраф за ранг (`r = 128`) и подготовка к `cert_version = 4` (FP8).
3. **[02. Глубокий анализ существующих майнеров](docs/02_Анализ_существующих_майнеров.md)** — сравнительный аудит коммерческих (`PeakMiner`, `SRBMiner-MULTI`, `ForgeMiner`, `KRig`, `OneZeroMiner`, `BzMiner`, `lpminer` и др.) и открытых GitHub-майнеров (`blackwell_pearl_miner`, `spark-pearl-miner`, `X-Miner`, `CPPminer` и др.).
4. **[03. Микроархитектура NVIDIA RTX 40 (`sm_89`) и RTX 50 (`sm_120`)](docs/03_Микроархитектура_RTX40_Ada_и_RTX50_Blackwell.md)** — физические пределы `IMMA.16832` (1024 MAC/такт/SM), 100% L2-резидентность базиса (`8.5 МиБ` при `M=N=32768`) и обход деградации `ptxas -O3`.
5. **[04. Концепция и архитектура собственного майнера `ApexPRL`](docs/04_Концепция_и_архитектура_собственного_майнера.md)** — детальная архитектура модулей и вычислительного конвейера.
6. **[05. Спецификация HiveOS Linux и приоритетного пула Kryptex (`Stratum v2 Gzip`)](docs/05_Спецификация_HiveOS_Linux_и_Kryptex_Stratum_v2.md)** — побитовый протокол Kryptex и спецификация переносимости под образы HiveOS.
