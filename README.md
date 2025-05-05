# Kokoro Kyrgyz TTS

Модель синтеза речи на кыргызском языке для Kokoro.

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/kokoro-kyrgyz-tts.git
cd kokoro-kyrgyz-tts
```

2. Создайте виртуальное окружение и установите зависимости:
```bash
conda create -n kokoro-kyrgyz python=3.8
conda activate kokoro-kyrgyz
pip install -r requirements.txt
```

3. Скачайте датасет:
```bash
# Создайте директорию для датасета
mkdir -p kyrgyz_dataset/audio_files

# Скачайте аудио файлы и metadata.json
# (Инструкции по скачиванию датасета будут добавлены позже)
```

## Структура проекта

```
kokoro-kyrgyz-tts/
├── train_kokoro_kyrgyz.py    # Скрипт обучения
├── requirements.txt          # Зависимости
├── README.md                # Документация
├── kyrgyz_dataset/          # Датасет
│   ├── audio_files/        # Аудио файлы
│   └── metadata.json       # Метаданные
└── .gitignore              # Исключения для git
```

## Обучение модели

```bash
python train_kokoro_kyrgyz.py
```

## Использование

После обучения модель будет сохранена в директории `kyrgyz-tts-model/`. Для использования с Kokoro:

```python
from kyrgyz_tts.kyrgyz_tts_interface import KyrgyzTTS
kokoro.register_tts("kyrgyz", KyrgyzTTS())
kokoro.speak("Ваш текст на кыргызском", language="kyrgyz")
```

## Требования

- Python 3.8+
- CUDA-совместимая видеокарта (рекомендуется)
- Минимум 16GB RAM
- Минимум 50GB свободного места на диске 