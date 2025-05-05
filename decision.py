import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
from loguru import logger
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate

# Настройка логирования
logger.add("kokoro_training.log", rotation="1 day")

# Конфигурация
class Config:
    # Пути
    DATASET_DIR = Path("kyrgyz_dataset")
    AUDIO_DIR = DATASET_DIR / "audio_files"
    METADATA_PATH = DATASET_DIR / "metadata.json"
    OUTPUT_DIR = Path("output")
    
    # Параметры аудио
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 3.0  # секунды
    
    # Параметры обучения
    MODEL_NAME = "openai/whisper-small"
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100
    NUM_EPOCHS = 2
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # Ограничение датасета
    MAX_SAMPLES = 10
    
    # Создание директорий
    OUTPUT_DIR.mkdir(exist_ok=True)

def load_dataset():
    """Загрузка и подготовка датасета."""
    logger.info("Загрузка датасета...")
    
    # Проверка существования директорий
    if not Config.DATASET_DIR.exists():
        raise FileNotFoundError(f"Директория с датасетом не найдена: {Config.DATASET_DIR}")
    if not Config.AUDIO_DIR.exists():
        raise FileNotFoundError(f"Директория с аудио не найдена: {Config.AUDIO_DIR}")
    if not Config.METADATA_PATH.exists():
        raise FileNotFoundError(f"Файл метаданных не найден: {Config.METADATA_PATH}")
    
    # Загрузка метаданных
    with open(Config.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    logger.info(f"Загружено {len(metadata)} записей из метаданных")
    
    # Создание списков для датасета
    audio_paths = []
    texts = []
    
    # Ограничение количества примеров
    max_samples = min(Config.MAX_SAMPLES, len(metadata))
    logger.info(f"Будет обработано максимум {max_samples} примеров")
    
    for i, item in enumerate(metadata[:max_samples]):  # Ограничиваем количество примеров
        try:
            audio_path = Config.AUDIO_DIR / item["audio_filename"]
            if audio_path.exists():
                audio_paths.append(str(audio_path))
                texts.append(item["transcription"])
            else:
                logger.warning(f"Аудиофайл не найден: {audio_path}")
        except KeyError as e:
            logger.error(f"Отсутствует ключ в метаданных, запись {i}: {e}")
            continue
    
    if not audio_paths:
        raise ValueError("Не найдено ни одного валидного аудиофайла")
    
    # Создание датасета
    dataset = Dataset.from_dict({
        "audio_path": audio_paths,
        "text": texts
    })
    
    logger.info(f"Создан датасет с {len(dataset)} примерами")
    return dataset

def process_audio(audio_path: str, processor) -> np.ndarray:
    """Обработка аудиофайла."""
    try:
        logger.debug(f"Обработка аудио: {audio_path}")
        
        # Загрузка аудио
        waveform, sample_rate = torchaudio.load(audio_path)
        logger.debug(f"Загружено аудио: {waveform.shape}, sample_rate={sample_rate}")
        
        # Конвертация в моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.debug(f"Конвертировано в моно: {waveform.shape}")
        
        # Ресемплинг
        if sample_rate != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=Config.SAMPLE_RATE
            )
            waveform = resampler(waveform)
            logger.debug(f"Ресемплинг до {Config.SAMPLE_RATE} Гц: {waveform.shape}")
        
        # Ограничение длины
        max_samples = int(Config.MAX_AUDIO_LENGTH * Config.SAMPLE_RATE)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
            logger.debug(f"Обрезано до {max_samples} сэмплов: {waveform.shape}")
        
        # Извлечение признаков
        input_features = processor(
            audio=waveform.squeeze().numpy(),
            sampling_rate=Config.SAMPLE_RATE,
            return_tensors="pt"
        ).input_features
        
        logger.debug(f"Извлечены признаки: {input_features.shape}")
        return input_features.squeeze().numpy().astype(np.float32)
    
    except Exception as e:
        logger.error(f"Ошибка при обработке аудио {audio_path}: {str(e)}")
        return None

def prepare_dataset(dataset: Dataset, processor) -> Dataset:
    """Подготовка датасета для обучения."""
    def process_batch(batch):
        processed_batch = {
            "input_features": [],
            "labels": []
        }
        
        for audio_path, text in zip(batch["audio_path"], batch["text"]):
            # Обработка аудио
            input_features = process_audio(audio_path, processor)
            if input_features is None:
                continue
            
            # Токенизация текста
            tokenized_text = processor.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=448,
                padding="max_length"
            )
            
            processed_batch["input_features"].append(input_features)
            processed_batch["labels"].append(tokenized_text["input_ids"].squeeze().numpy().astype(np.int32))
        
        return processed_batch
    
    # Обработка датасета
    processed_dataset = dataset.map(
        process_batch,
        remove_columns=["audio_path", "text"],
        batched=True,
        batch_size=Config.BATCH_SIZE,
        desc="Обработка датасета"
    )
    
    # Фильтрация пустых примеров
    processed_dataset = processed_dataset.filter(lambda x: x is not None)
    logger.info(f"Обработано {len(processed_dataset)} примеров")
    
    return processed_dataset

def train_model(dataset: Dataset):
    """Обучение модели Whisper."""
    try:
        logger.info("Инициализация модели...")
        
        # Загрузка модели и процессора
        processor = WhisperProcessor.from_pretrained(Config.MODEL_NAME)
        model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_NAME)
        
        # Добавление поддержки кыргызского языка
        tokenizer = processor.tokenizer
        if "<|ky|>" not in tokenizer.additional_special_tokens:
            special_tokens = tokenizer.additional_special_tokens
            special_tokens.append("<|ky|>")
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            model.resize_token_embeddings(len(tokenizer))
        
        # Подготовка датасета
        processed_dataset = prepare_dataset(dataset, processor)
        
        # Разделение на train/val
        train_val_split = processed_dataset.train_test_split(test_size=0.1)
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]
        
        # Настройка обучения
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(Config.OUTPUT_DIR),
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            learning_rate=Config.LEARNING_RATE,
            warmup_steps=Config.WARMUP_STEPS,
            num_train_epochs=Config.NUM_EPOCHS,
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            logging_dir=str(Config.OUTPUT_DIR / "logs"),
            logging_steps=10,
            report_to="tensorboard",
            fp16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            dataloader_num_workers=2,
            remove_unused_columns=False
        )
        
        # Метрика WER
        wer_metric = evaluate.load("wer")
        
        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            
            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
            
            wer = wer_metric.compute(predictions=pred_str, references=label_str)
            return {"wer": wer}
        
        # Инициализация тренера
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor.feature_extractor,
            compute_metrics=compute_metrics
        )
        
        # Обучение
        logger.info("Начало обучения...")
        trainer.train()
        
        # Сохранение модели
        logger.info(f"Сохранение модели в {Config.OUTPUT_DIR}")
        model.save_pretrained(Config.OUTPUT_DIR)
        processor.save_pretrained(Config.OUTPUT_DIR)
        
        logger.info("Обучение успешно завершено")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise

def main():
    """Основная функция."""
    try:
        # Загрузка датасета
        dataset = load_dataset()
        
        # Обучение модели
        train_model(dataset)
        
    except Exception as e:
        logger.error(f"Ошибка в main: {str(e)}")
        raise
if __name__ == "__main__":
    main()
