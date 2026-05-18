# LightWeight MAPF-GPT — `extras/`

Дополнения к [CognitiveAISystems/MAPF-GPT](https://github.com/CognitiveAISystems/MAPF-GPT):
**сжатие модели** (trim, dynamic int8, AWQ, SmoothQuant), **дистилляция**
с hidden-state-сигналом и multi-position-таргетами, единый **бенчмарк POGEMA**
и **чистая latency-бенчмарка**.

Все экспериментальные артефакты пишутся в `extras/runs/<category>/<timestamp>_<tag>/`.

## Структура

```
extras/
  lmgpt/                       # питон-пакет с переиспользуемой логикой
    checkpoints.py             # I/O чекпойнтов
    compression/               # trim, dynamic_int8, awq, smoothquant
    distillation/              # data, losses, trainer, finetune
    eval/                      # POGEMA harness, comparison tables, latency
    utils/                     # env capture, jsonl, run-dirs
  scripts/                     # тонкие CLI-обёртки над lmgpt
  configs/
    students/                  # JSON-конфиги студентов (GPTConfig fields)
    distill/                   # JSON-конфиги тренировки дистилляции
    awq/                       # конфиги квантизации
  runs/                        # все эксперименты пишут сюда (в .gitignore)
    baselines/  trim/  trim_ft/  distill/  awq/  latency/  reports/
  integration/                 # эталонные правки upstream-кода (inference, example)
```

## Среда

Все скрипты ожидают активного окружения `mapf-gpt`:

```bash
source /data/conda/miniconda3/etc/profile.d/conda.sh
conda activate mapf-gpt
cd /data/homes/makarov_vd/workspace/CCM/MAPF-GPT
```

PyTorch 2.10 + CUDA 12.8 в этом окружении уже есть. Опциональные
зависимости (`torchao`, `bitsandbytes`) ставятся отдельно.

### Comet ML (обучение)

Дистилляция и finetune **по умолчанию логируют метрики в Comet** (графики
`train_batch_loss`, `val_*`, `rollout_*`, `lr`). Ключ: `extras/.key_comet`
(первая строка) или переменная `COMET_API_KEY`. Проект по умолчанию:
`LightWeight-MAPF-GPT` (переопределение: `COMET_PROJECT_NAME` или
`--comet-project`). Отключить: `--no-comet`.

```bash
pip install comet-ml
```

## Что → каким скриптом

| Что нужно сделать                                            | CLI                                                                 |
|--------------------------------------------------------------|---------------------------------------------------------------------|
| Срезать слои у чекпойнта                                     | `extras/scripts/trim.py`                                            |
| Дообучить срезанный чекпойнт (CE или CE+KL self-distill)     | `extras/scripts/finetune_trimmed.py`                                |
| Дистиллировать в маленького студента (вкл. hidden+multipos)  | `extras/scripts/train_distill.py`                                   |
| Квантовать в W4A16 (AWQ) под GPU                             | `extras/scripts/quantize_awq.py`                                    |
| SmoothQuant-preprocessing (W8A8 quality ablation)            | `extras/scripts/quantize_smoothquant.py`                            |
| Dynamic int8 (CPU-only baseline / antipattern)               | `extras/scripts/quantize_dynamic_int8.py`                           |
| Прогнать POGEMA benchmark (авторские веса или custom)        | `extras/scripts/eval_pogema.py`                                     |
| Собрать таблицы сравнения нескольких run-ов                  | `extras/scripts/compare_runs.py`                                    |
| Замерить чистую forward-latency без среды                    | `extras/scripts/bench_latency.py`                                   |

Подробности — внутри каждого скрипта (`--help`) и в докстрингах модулей пакета.

## Типичный конвейер (полный цикл)

```bash
# 1. Бейзлайны (2M, 6M, 85M) — на полном POGEMA, FP32
python extras/scripts/eval_pogema.py --models 2M 6M 85M \
  --output-root extras/runs/baselines

# 2. Trim + fine-tune (CE+KL self-distillation)
python extras/scripts/trim.py -i weights/model-2M.pt -o weights/model-2M-L4.pt --n_layer 4
python extras/scripts/finetune_trimmed.py \
  --trimmed weights/model-2M-L4.pt --teacher weights/model-2M.pt \
  --mode ce_kl --out-dir extras/runs/trim_ft/L4_cekl
python extras/scripts/eval_pogema.py \
  --custom-weights extras/runs/trim_ft/2026-..._L4_cekl/ckpt_best.pt \
  --custom-model-name 2M-L4-cekl \
  --output-root extras/runs/trim_ft

# 3. Дистилляция (KL+CE + hidden-state + multipos)
python extras/scripts/train_distill.py \
  --config extras/configs/distill/with_hidden.json \
  --student extras/configs/students/student-2M.json \
  --teacher weights/model-85M.pt \
  --out-dir extras/runs/distill/2M_from_85M_hidden
python extras/scripts/eval_pogema.py \
  --custom-weights extras/runs/distill/2026-..._2M_from_85M_hidden/ckpt_best.pt \
  --custom-model-name student-2M-distilled-85M-hidden \
  --output-root extras/runs/distill

# 4. AWQ-квантизация
python extras/scripts/quantize_awq.py \
  -i weights/model-2M.pt -o weights/model-2M-awq.pt \
  --calib-data dataset/validation --calib-batches 256 --calib-max-files 4
python extras/scripts/eval_pogema.py \
  --custom-weights weights/model-2M-awq.pt \
  --custom-model-name 2M-AWQ-W4A16 \
  --output-root extras/runs/awq

# 5. Чистая latency-бенчмарка
python extras/scripts/bench_latency.py \
  --weights weights/model-2M.pt --label 2M_FP32

# 6. Сводный отчёт
python extras/scripts/compare_runs.py \
  --runs extras/runs/baselines/2026-... \
  extras/runs/trim_ft/2026-... \
  extras/runs/distill/2026-... \
  extras/runs/awq/2026-...
```

## Соглашения

- **Корень запуска**: все CLI выполняются из корня репозитория (`MAPF-GPT/`).
- **Output layout**: `<out_dir_base_parent>/<YYYY-MM-DD_HH-MM-SS>_<basename>/...` —
  внутри лежат `ckpt_*.pt`, `metrics.jsonl`, `summary_latest.json`, `run_config.json`,
  `environment.txt` (git/python/torch/nvidia-smi), `errors.log`, опциональный `README.md`.
- **POGEMA метрики**: `CSR`, `ISR`, `SoC`, `ep_length`, `runtime` (end-to-end). См.
  [POGEMA paper](https://arxiv.org/abs/2407.14931).

## Зависимости (опциональные)

```bash
pip install comet-ml  # логирование обучения (train_distill, finetune_trimmed)
pip install torchao   # W4/W8 GPU kernels (необходим для AWQ-инференса)
```

## Финальные эксперименты

Полный отчёт (POGEMA + latency по всем чекпойнтам, разбивка по `num_agents`):
**[REPORT.md](REPORT.md)**.

Сводки: `extras/runs/reports/final_2026-05-17/` (`global_by_model.md`,
`by_map_group_agents.md`, `latency_b2048.md`).

Частичные прогоны (smoke / 2 группы POGEMA) удалены. Запускайте этапы из
раздела «Типичный конвейер» с **полным** eval (без `--groups`) и полным
обучением (без `--max-train-files`).

### Веса (не в git)

Список канонических чекпойнтов: `extras/weights_manifest.json`. Упаковать zip
для передачи коллегам:

```bash
bash extras/scripts/package_weights.sh v1
# -> dist/lightweight-checkpoints-v1.zip (~200–250 MB)
```

Author **2M/6M/85M** — как в [upstream README](https://github.com/JohnSili/MAPF-GPT)
(Hugging Face). В zip только наши обученные модели.
