import os
import torch
import torchaudio
import numpy as np
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from loguru import logger
import json
from pathlib import Path

# Настройки
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Параметры обучения
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 4

def process_dataset(dataset: Dataset, processor) -> Dataset:
    """
    Обработка датасета с использованием процессора Whisper.
    """
    def prepare_dataset(batch):
        try:
            # Загрузка аудио
            audio_paths = batch["audio_path"]
            texts = batch["text"]
            
            processed_batch = {
                "input_features": [],
                "labels": []
            }
            
            for audio_path, text in zip(audio_paths, texts):
                try:
                    # Загрузка аудиофайла
                    waveform, sample_rate = torchaudio.load(audio_path)
                    
                    # Конвертация в моно, если необходимо
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Ресемплинг до 16000 Гц, если необходимо
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=sample_rate,
                            new_freq=16000
                        )
                        waveform = resampler(waveform)
                        sample_rate = 16000
                    
                    # Ограничиваем длину аудио до 1 секунды
                    max_samples = int(1 * sample_rate)
                    if waveform.shape[1] > max_samples:
                        waveform = waveform[:, :max_samples]
                    
                    # Извлечение признаков аудио
                    input_features = processor(
                        audio=waveform.squeeze().numpy(),
                        sampling_rate=sample_rate,
                        return_tensors="pt"
                    ).input_features
                    
                    # Токенизируем текст
                    tokenized_text = processor.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=448,
                        padding="max_length"
                    )
                    
                    # Создаем labels из токенизированного текста
                    labels = tokenized_text["input_ids"].squeeze()
                    
                    # Добавляем в батч
                    processed_batch["input_features"].append(input_features.squeeze().numpy().astype(np.float16))
                    processed_batch["labels"].append(labels.numpy().astype(np.int16))
                    
                except Exception as e:
                    logger.error(f"Ошибка при обработке аудиофайла {audio_path}: {str(e)}")
                    continue
            
            if not processed_batch["input_features"] or not processed_batch["labels"]:
                return None
                
            return processed_batch
        except Exception as e:
            logger.error(f"Ошибка при обработке батча: {str(e)}")
            return None
    
    # Обрабатываем датасет
    processed_dataset = dataset.map(
        prepare_dataset,
        remove_columns=["audio_path", "text"],
        desc="Обработка датасета",
        batched=True,
        batch_size=1
    )
    
    # Фильтруем None значения
    processed_dataset = processed_dataset.filter(lambda x: x is not None)
    
    logger.info(f"Обработано {len(processed_dataset)} примеров")
    
    return processed_dataset

def fine_tune_whisper(dataset: Dataset, output_dir: str):
    """
    Дообучение модели Whisper на кыргызском языке.
    """
    try:
        logger.info("Загрузка модели Whisper...")
        # Загружаем предобученную модель Whisper
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        
        # Добавляем поддержку кыргызского языка
        tokenizer = processor.tokenizer
        if "<|ky|>" not in tokenizer.additional_special_tokens:
            special_tokens = tokenizer.additional_special_tokens
            special_tokens.append("<|ky|>")
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            model.resize_token_embeddings(len(tokenizer))
        
        # Обработка датасета
        processed_dataset = process_dataset(dataset, processor)
        
        # Разделение на train/val
        train_val_split = processed_dataset.train_test_split(test_size=0.1)
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]
        
        # Подготовка обучения
        from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
        import evaluate
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,  # Уменьшаем размер батча
            per_device_eval_batch_size=1,   # Уменьшаем размер батча
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_EPOCHS,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            report_to="tensorboard",
            fp16=True,  # Используем смешанную точность для экономии памяти
            gradient_checkpointing=True,  # Включаем gradient checkpointing
            optim="adafactor",  # Используем более эффективный оптимизатор
            dataloader_num_workers=0,  # Отключаем многопоточную загрузку данных
            remove_unused_columns=False,  # Не удаляем неиспользуемые колонки
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
            compute_metrics=compute_metrics,
        )
        
        # Обучение
        logger.info("Начало обучения...")
        trainer.train()
        
        # Сохранение модели
        logger.info(f"Сохранение модели в {output_dir}")
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        logger.info("Обучение успешно завершено")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise

if __name__ == "__main__":
    # Загрузка датасета
    # Загрузка метаданных
    metadata_path = Path("kyrgyz_dataset/metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Создание списков путей и текстов
    audio_paths = []
    texts = []
    
    for item in metadata:
        audio_path = Path("kyrgyz_dataset/audio_files") / item["audio_filename"]
        if audio_path.exists():
            audio_paths.append(str(audio_path))
            texts.append(item["transcription"])
    
    # Создание датасета
    dataset = Dataset.from_dict({
        "audio_path": audio_paths,
        "text": texts
    })
    
    logger.info(f"Загружено {len(dataset)} примеров")
    
    # Дообучение модели
    fine_tune_whisper(dataset, OUTPUT_DIR)