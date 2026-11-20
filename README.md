# Разработка высокоэффективного майнера Pearl (PRL) — `ApexPRL`

Рабочая директория научно-исследовательской разработки и полного исходного кода собственного специализированного GPU-майнера криптовалюты **Pearl (`PRL`, алгоритм `PearlHash` / `cuPOW` / `NoisyGEMM`)** под видеокарты **NVIDIA GeForce RTX 40-й серии (`sm_89` Ada Lovelace)** и **RTX 50-й серии (`sm_120a` Blackwell)** с приоритетом под **HiveOS Linux** и пул **Kryptex (`Stratum v2 Gzip`)**.

- **Версия:** `v0.2.0` (**Hyper-L2 Direct 2-CTA/SM Engine**)
- **Комиссия разработчика (Dev Fee):** `0.00%` (полностью отсутствует).
- **Целевая ОС:** **HiveOS Linux** / любой дистрибутив Linux `x86_64` с `GLIBC >= 2.30` (совместимо с Ubuntu 20.04 / 22.04 / 24.04) и драйвером NVIDIA `>= 570.x` (для RTX 50) или `>= 535.x` (для RTX 40).
- **Приоритетный пул:** **Kryptex Pool** (`stratum+ssl://prl-ru.kryptex.network:8048`, резерв `stratum+ssl://prl-eu.kryptex.network:8048`) — нативный TLS (`rustls`), сжатие доказательств `gzip` (>20–50× экономии трафика на матрицах `Constant-64`), расчёт `compact_floor` и авто-подбор оптимальной статической сложности `d=...`.

---

## 1. Готовый релизный пакет для HiveOS Linux (`dist/`)

| Файл | Назначение | SHA-256 |
|---|---|---|
| [`dist/apexprl-hiveos.tar.gz`](dist/apexprl-hiveos.tar.gz) | Готовый архив **Custom Miner** для установки на риги HiveOS (`v0.2.0`) | `ff57545719bba25a4dd17344ecad4c01d551f10e011db3c9a9e93ff56d9f6c24` |
| [`dist/apexprl`](dist/apexprl) | Автономный переносимый Linux ELF-бинарник (`GLIBC_2.30`, встроенные `cubin` `Direct-L2` + `Basis` под `sm_89` и `sm_120a`) | `cbe6ecb6dcec90178265faf4eafbaf4c85838970721d1043debe564786fa276a` |
| [`dist/flight-sheet-kryptex.json`](dist/flight-sheet-kryptex.json) | Готовый шаблон полётного листа (Flight Sheet) HiveOS под пул Kryptex (`v0.2.0`) | — |
| [`dist/SHA256SUMS.txt`](dist/SHA256SUMS.txt) | Контрольные суммы релизных артефактов | — |

### Установка в HiveOS (Custom Miner + Kryptex)

1. **Прямая ссылка для поля Installation URL в HiveOS (`v0.2.0`):**
   ```text
   https://github.com/lunin02118-ops/apexprl/releases/download/v0.2.0/apexprl-hiveos.tar.gz
   ```
   *(Либо ручная установка по SSH на риге: `cd /tmp && wget https://github.com/lunin02118-ops/apexprl/releases/download/v0.2.0/apexprl-hiveos.tar.gz && tar -xzf apexprl-hiveos.tar.gz -C /hive/miners/custom/`)*
2. В веб-панели HiveOS создайте **Flight Sheet (Полётный лист)**:
   - **Coin:** `PRL`
   - **Wallet:** ваш адрес `prl1...` или идентификатор Kryptex (`krxXXXXXXX`)
   - **Pool:** `Configure in miner`
   - **Miner:** `Custom` -> `Setup Miner Config`:
     - **Miner name:** `apexprl`
     - **Installation URL:** `https://github.com/lunin02118-ops/apexprl/releases/download/v0.2.0/apexprl-hiveos.tar.gz`
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
       *(Внимание: в `v0.2.0` флаг `--lock-mem-idle` автоматически распознаёт внешний лок памяти **PenguinBurner / HiveOS OC** `810 МГц`, сохраняет его без перезаписи и не сбрасывает при остановке майнера!)*

---

## 2. Что нового в `ApexPRL v0.2.0` — Движок `Hyper-L2 Direct 2-CTA/SM`

По итогам боевого A/B-теста `v0.1.0` на риге **2 × RTX 5070 Ti (`70 SM`, `48 МиБ L2`, `2887 / 810 МГц`)** и глубокого SASS-разбора `apexprl_sm120a.cubin` в `v0.2.0` был разработан новый вычислительный движок **`--engine direct`** (активен по умолчанию):

1. **Полное удаление динамической сборки базиса из внутреннего `K`-цикла (`O(M·N·K)`):**
   - Вместо тяжёлой пошаговой сборки (`LDSM + IADD3 + PRMT + STS.128`) и 255-регистровых спиллов (`STL/LDL` в локальную память поверх `810 МГц` GDDR7) матрицы пред-материализуются в L2-резидентные свиззл-панели `K_STAGE = 64` через SWAR-ядро (`0x40404040u` + warp-shuffle транспонирование, `35–39` регистров, `0` спиллов).
   - Прямоугольная геометрия `M >= N` (`32768 × 8192 × 4096` на `48 МиБ L2` RTX 5070 / 5070 Ti; `32768 × 4096 × 4096` на `32 МиБ L2` RTX 4060; `32768 × 16384 × 4096` на `96 МиБ L2` RTX 5090): матрица `B` (`32 МиБ`) строится **один раз на блок** (`0.00%` оверхеда на раунд) и на 100% живёт в L2-кэше вместе с волново-выровненной панелью `A` (`panel_m = 13` тайлов -> `99.05%` wave efficiency на 70 SM RTX 5070 Ti!).
2. **2 CTA на каждый SM (`128×128`, `4 варпа/CTA = 8 варпов/SM`, `41 КиБ SMEM × 2`, `0` спиллов):**
   - Строгий аудит SASS (`tools/check_sass.py`): **`STL/LDL = 0`** во всех 6 ядрах `apexprl_direct_sm89.cubin` (`-O3`, `206–218` рег.) и `apexprl_direct_sm120a.cubin` (`-O1`, `201–210` рег.).
   - Два независимых CTA на SM полностью скрывают задержки `BAR.SYNC`, `TMA`/`cp.async` и эпилога `BLAKE3`.
   - На Blackwell (`sm_120a`) используется одноинструкционный `elect.sync` TMA (`UBLKCP`, `0` циклов `BRA.U`) и вариант `hl2-2cta-df` с отложенным `zero-RAW` сворачиванием (`FZ`).
3. **Устранение узкого места PCIe и полная совместимость с `PenguinBurner`:**
   - В `direct`-движке полностью убран по-тайловый uncached опрос `abort_epoch_ptr` через PCIe-райзеры (раунд длится всего `~5–12 мс`, прерывание проверяется между панелями на хосте).
   - В `nvml_drv.rs` сохраняется уже активный внешний лок памяти (`700–1050 МГц`, включая `810 МГц` от PenguinBurner) и никогда не вызывается `nvmlDeviceResetMemoryLockedClocks` для чужих настроек при выходе.
4. **Верифицированные результаты на живом кремнии (`RTX 4060 Laptop`, `sm_89`, `24 SM`, `32 МиБ L2`):**
   - **Самотест Gate G0 (`--self-test`):** 100% побитовое совпадение GPU == эталонный CPU == официальный верификатор `zk-pow` V3 + `gzip` одновременно для **`direct-l2`** (`105/105` попаданий, все 3 варианта, `21.3×` сжатие) и **`basis-160`** (`242/242` попаданий, все 4 варианта, `20.4×` сжатие).
   - **Эффективность от теоретического пика железа (`SOL = SMs × 1024 × f_SM`):** рост с **`86.4%` SOL** (`v0.1.0 basis`) до **`91.7%` SOL** (`v0.2.0 direct --k 4096`) и **`92.5%` SOL** (`v0.2.0 direct --k 2048`) на Ada Lovelace (`sm_89`), с ещё большим приростом на Blackwell (`sm_120a`) за счёт снятия 16B-спиллов, удаления пересборки базиса и аппаратного `elect.sync` TMA.

---

## 3. Прямой запуск из командной строки Linux

```bash
# 1. Список обнаруженных видеокарт NVIDIA и их параметров SM / L2
./dist/apexprl --list-devices

# 2. Двойной аппаратный самотест (Direct-L2 + Basis-160: GPU == CPU == официальный zk-pow V3 + gzip)
./dist/apexprl --self-test

# 3. Автономный бенчмарк Hyper-L2 Direct (v0.2, по умолчанию) с автотюнингом и расчётом % от SOL
./dist/apexprl --benchmark 10 --tune

# (Опционально: сравнить с K=2048 или старым движком v0.1 --engine basis)
./dist/apexprl --benchmark 10 --tune --k 2048
./dist/apexprl --benchmark 10 --tune --engine basis

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

- [`Cargo.toml`](Cargo.toml) — корень Rust-workspace (`v0.2.0`):
  - [`crates/apexprl-pow/`](crates/apexprl-pow/src/lib.rs) — математика консенсуса `PearlHash V3` (Salted Seed), единый кэш дерева Меркла для прямоугольных (`M >= N`) и квадратных (`M == N`) `Constant-64` матриц, сборщик `PlainProof`, встроенная пред-отправочная верификация через официальный `zk-pow` и `gzip`-сжатие.
  - [`crates/apexprl-gpu/`](crates/apexprl-gpu/src/lib.rs) — динамические `dlopen`-обёртки над `libcuda.so.1` и `libnvidia-ml.so.1`, движки `DirectL2` (волновое выравнивание панелей `optimal_panel_m_tiles` + авто-размер `N` под L2) и `Basis160`, автотюнер вариантов (`--tune`), бережное NVML-управление с сохранением внешнего OC (PenguinBurner).
  - [`crates/apexprl-stratum/`](crates/apexprl-stratum/src/lib.rs) — сетевой стек `rustls` TLS / TCP для `Kryptex Stratum v2 (gzip)` и классического `Stratum v1` с автоматическим Failover.
  - [`crates/apexprl-cli/`](crates/apexprl-cli/src/main.rs) — мульти-GPU рантайм, пул потоков сборки доказательств, бенчмарк с живым контролем `avgSM` / `% SOL` и HTTP-сервер телеметрии [`api_server.rs`](crates/apexprl-cli/src/api_server.rs).
- [`kernels/`](kernels/) — специализированные CUDA C++20 / PTX вычислительные ядра:
  - [`kernels/apexprl_direct_sm120a.cu`](kernels/apexprl_direct_sm120a.cu) — ядро **v0.2 Hyper-L2 Direct под RTX 50 (`sm_120a`)**: 2 CTA/SM, `elect.sync` TMA (`UBLKCP`), `0` спиллов (`-Xptxas -O1`), 3 варианта (`hl2-2cta`, `hl2-2cta-df`, `hl2-3stg`).
  - [`kernels/apexprl_direct_sm89.cu`](kernels/apexprl_direct_sm89.cu) — ядро **v0.2 Hyper-L2 Direct под RTX 40 (`sm_89`)**: 2 CTA/SM, `cp.async.cg`, `0` спиллов (`-Xptxas -O3`), 3 варианта (`hl2-2cta`, `hl2-2cta-s2`, `hl2-3stg`).
  - [`kernels/common/direct_l2_core.cuh`](kernels/common/direct_l2_core.cuh) и [`direct_l2_params.cuh`](kernels/common/direct_l2_params.cuh) — SWAR warp-transpose материализатор `apx_materialize_panel_128` и параметризованные Direct-L2 GEMM-ядра.
  - [`kernels/apexprl_sm120a.cu`](kernels/apexprl_sm120a.cu) & [`kernels/apexprl_sm89.cu`](kernels/apexprl_sm89.cu) — резервные ядра `Basis-160x256` (`--engine basis`).
- [`hiveos/`](hiveos/) — скрипты интеграции с HiveOS (`h-manifest.conf`, `h-config.sh`, `h-run.sh`, `h-stats.sh`, `flight-sheet-kryptex.json`).
- [`scripts/package_hiveos.sh`](scripts/package_hiveos.sh) и [`tools/check_sass.py`](tools/check_sass.py) — автоматизированный SASS-аудит всех 4 `cubin` (строго `0` спиллов на Direct-L2), сборка через `cargo zigbuild --target x86_64-unknown-linux-gnu.2.31` и упаковка `dist/apexprl-hiveos.tar.gz`.

---

## 5. Навигация по аналитической и архитектурной документации

1. **[Сводный итоговый отчёт НИОКР](ОТЧЕТ_ИССЛЕДОВАНИЕ_И_АРХИТЕКТУРА_МАЙНЕРА_PEARL.md)** — главный обзорный документ с таблицами сравнения майнеров и расчётами энергоэффективности.
2. **[01. Исследование экосистемы и алгоритма Pearl (`cuPOW`)](docs/01_Исследование_экосистемы_и_алгоритма_Pearl.md)** — математическая модель `NoisyGEMM`, `cert_version = 3` (Salted Seed), штраф за ранг (`r = 128`) и подготовка к `cert_version = 4` (FP8).
3. **[02. Глубокий анализ существующих майнеров](docs/02_Анализ_существующих_майнеров.md)** — сравнительный аудит коммерческих и открытых майнеров.
4. **[03. Микроархитектура NVIDIA RTX 40 (`sm_89`) и RTX 50 (`sm_120`)](docs/03_Микроархитектура_RTX40_Ada_и_RTX50_Blackwell.md)** — физические пределы `IMMA.16832` (1024 MAC/такт/SM), иерархия L2-кэша и правила компиляции `ptxas`.
5. **[04. Концепция и архитектура собственного майнера `ApexPRL`](docs/04_Концепция_и_архитектура_собственного_майнера.md)** — детальная архитектура модулей и вычислительного конвейера.
6. **[05. Спецификация HiveOS Linux и приоритетного пула Kryptex (`Stratum v2 Gzip`)](docs/05_Спецификация_HiveOS_Linux_и_Kryptex_Stratum_v2.md)** — побитовый протокол Kryptex и спецификация переносимости под образы HiveOS.
