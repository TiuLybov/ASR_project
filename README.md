# ASR Homework — Listen, Attend and Spell

Реализация модели Listen, Attend and Spell (LAS) для автоматического распознавания речи на датасете LibriSpeech.

## Архитектура

**Listener** — пирамидальный BiLSTM энкодер. Первый слой — обычный BiLSTM, далее 3 слоя pBLSTM, каждый из которых уменьшает временное разрешение в 2 раза за счёт конкатенации соседних фреймов. Итого сжатие по времени в 8 раз.

**Attention** — content-based (Bahdanau) attention между decoder hidden state и encoder выходами.

**Speller** — двухслойный LSTM декодер с attention. На каждом шаге принимает embedding предыдущего токена и контекст от attention, выдаёт распределение по словарю.

## Установка

```bash
git clone https://github.com/<your_username>/asr_project.git
cd asr_project
pip install -r requirements.txt
```

## Скачивание весов

```bash
# скачать веса финальной модели (заменить на свою ссылку)
pip install gdown
gdown <GOOGLE_DRIVE_LINK> -O saved/model_best.pth
```

## Обучение

### One-Batch Test

```bash
python train.py datasets=onebatchtest trainer.n_epochs=100 trainer.epoch_len=10 dataloader.batch_size=2 writer.mode=offline trainer.override=True
```

### Обучение на train-clean-100

```bash
python train.py
```

### Обучение с другим конфигом

```bash
python train.py datasets=example model=las trainer.n_epochs=100
```

### Обучение без аугментаций

```bash
python train.py transforms/instance_transforms=no_augs
```

## Инференс

```bash
python inference.py \
  inferencer.from_pretrained=saved/model_best.pth \
  datasets=example_eval \
  inferencer.save_path=inference_output
```

Предсказания сохраняются в `data/saved/inference_output/<partition>/<utterance_id>.txt`.

### Инференс на своих данных

Подготовьте директорию в формате:
```
my_data/
├── audio/
│   ├── utt1.wav
│   └── utt2.wav
└── transcriptions/  (опционально)
    ├── utt1.txt
    └── utt2.txt
```

```bash
python inference.py \
  datasets=custom_dir \
  datasets.test.audio_dir=my_data/audio \
  datasets.test.transcription_dir=my_data/transcriptions \
  inferencer.from_pretrained=saved/model_best.pth
```

## Подсчёт метрик

```bash
python calc_metrics.py \
  --gt-dir my_data/transcriptions \
  --pred-dir data/saved/inference_output/test
```

## Аугментации

Реализовано 4 аугментации:

1. **SpeedPerturb** — изменение скорости воспроизведения (0.9–1.1x) через линейную интерполяцию
2. **Gain** — случайное изменение громкости (через torch_audiomentations)
3. **AdditiveNoise** — добавление гауссовского шума к waveform
4. **FrequencyMasking** + **TimeMasking** — маскирование частотных полос и временных отрезков на спектрограмме (SpecAugment)

Wav-аугментации применяются ДО вычисления спектрограммы, spec-аугментации — ПОСЛЕ.

## Beam Search

Реализован hand-crafted beam search в `src/text_encoder/ctc_text_encoder.py`. Метрики с beam search (beam_size=5) логируются при валидации наряду с greedy.

## Структура проекта

```
├── train.py                    # скрипт обучения
├── inference.py                # скрипт инференса
├── calc_metrics.py             # подсчёт WER/CER
├── requirements.txt
├── src/
│   ├── configs/                # Hydra конфиги
│   ├── datasets/               # датасеты, collate, утилиты
│   ├── loss/                   # LAS cross-entropy loss
│   ├── metrics/                # CER, WER (greedy + beam search)
│   ├── model/                  # LAS модель (Listener + Attention + Speller)
│   ├── text_encoder/           # кодировщик текста с SOS/EOS/PAD
│   ├── trainer/                # Trainer и Inferencer
│   ├── transforms/             # wav и spec аугментации
│   ├── logger/                 # W&B / CometML логгеры
│   └── utils/
```

## W&B логирование

Логируются:
- Loss на каждом шаге
- Learning rate
- Gradient norm
- Спектрограммы (train/val)
- Таблица предсказаний (target, greedy prediction, beam search prediction, CER, WER)

## Воспроизведение результатов

1. Обучение baseline: 50 эпох на train-clean-100 с конфигом по умолчанию
2. Дообучение: понижение lr, увеличение epoch_len
3. Для test-other: дообучение на train-clean-360 + аугментации
