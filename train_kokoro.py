import torch
import torchaudio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Audio
import numpy as np
from loguru import logger
from pathlib import Path
import os

# Настройки
class Config:
    MODEL_NAME = "openai/whisper-small"
    DATASET_DIR = Path("kyrgyz_dataset")
    OUTPUT_DIR = Path("output")
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 30.0  # секунды
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-5

def load_dataset():
    """Загрузка и подготовка датасета."""
    logger.info("Загрузка датасета...")
    
    # Создаем список аудио файлов и транскрипций
    audio_files = []
    transcriptions = []
    
    # Читаем метаданные
    metadata_path = Config.DATASET_DIR / "metadata.csv"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                audio_file, text = line.strip().split("|")
                audio_path = Config.DATASET_DIR / "audio_files" / audio_file
                if audio_path.exists():
                    audio_files.append(str(audio_path))
                    transcriptions.append(text)
    
    # Создаем датасет
    dataset = {
        "audio": audio_files,
        "text": transcriptions
    }
    
    return dataset

def prepare_dataset(dataset):
    """Подготовка датасета для обучения."""
    logger.info("Подготовка датасета...")
    
    # Загрузка аудио
    def load_audio(batch):
        try:
            # Загрузка аудио
            waveform, sample_rate = torchaudio.load(batch["audio"])
            
            # Конвертация в моно
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Ресемплинг
            if sample_rate != Config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=Config.SAMPLE_RATE
                )
                waveform = resampler(waveform)
            
            # Ограничение длины
            max_samples = int(Config.MAX_AUDIO_LENGTH * Config.SAMPLE_RATE)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            
            return {
                "audio": waveform.squeeze().numpy(),
                "sampling_rate": Config.SAMPLE_RATE
            }
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио: {str(e)}")
            return None
    
    # Применяем функцию загрузки аудио
    processed_dataset = []
    for i in range(len(dataset["audio"])):
        result = load_audio({"audio": dataset["audio"][i]})
        if result is not None:
            processed_dataset.append({
                "audio": result["audio"],
                "sampling_rate": result["sampling_rate"],
                "text": dataset["text"][i]
            })
    
    return processed_dataset

def train_model(dataset):
    """Обучение модели."""
    logger.info("Начало обучения...")
    
    # Загрузка модели и токенизатора
    processor = WhisperProcessor.from_pretrained(Config.MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_NAME)
    
    # Подготовка данных
    def prepare_example(example):
        # Извлечение признаков
        input_features = processor(
            audio=example["audio"],
            sampling_rate=example["sampling_rate"],
            return_tensors="pt"
        ).input_features
        
        # Токенизация текста
        labels = processor(
            text=example["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=448
        ).input_ids
        
        return {
            "input_features": input_features.squeeze(),
            "labels": labels.squeeze()
        }
    
    # Применяем подготовку данных
    train_dataset = [prepare_example(example) for example in dataset]
    
    # Создаем коллатор данных
    data_collator = DataCollatorForSeq2Seq(
        processor=processor,
        model=model
    )
    
    # Настройки обучения
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(Config.OUTPUT_DIR),
        per_device_train_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=500,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=Config.BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )
    
    # Создаем тренер
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor,
    )
    
    # Обучение
    trainer.train()
    
    # Сохранение модели
    trainer.save_model(str(Config.OUTPUT_DIR))
    processor.save_pretrained(str(Config.OUTPUT_DIR))
    
    logger.info("Обучение завершено!")

def main():
    """Основная функция."""
    try:
        # Создаем директории
        Config.OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Загрузка и подготовка датасета
        dataset = load_dataset()
        processed_dataset = prepare_dataset(dataset)
        
        # Обучение модели
        train_model(processed_dataset)
        
    except Exception as e:
        logger.error(f"Ошибка в main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 