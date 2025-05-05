# from datasets import load_dataset
# import os
# import json
# import soundfile as sf
# from tqdm import tqdm

# # Создаем необходимые директории
# os.makedirs('kyrgyz_dataset/audio_files', exist_ok=True)

# # Загружаем датасет
# ds = load_dataset("Simonlob/Kany_dataset_mk4_Base", split='train[:1000]')

# # Список для хранения метаданных
# metadata = []

# # Обрабатываем каждый пример в датасете
# for idx, example in enumerate(tqdm(ds, desc="Обработка датасета")):
#     # Получаем аудио данные
#     audio_data = example['audio']['array']
#     sample_rate = example['audio']['sampling_rate']
    
#     # Создаем имя файла
#     audio_filename = f"audio_{idx}.wav"
#     audio_path = os.path.join('kyrgyz_dataset/audio_files', audio_filename)
    
#     # Сохраняем аудио файл
#     sf.write(audio_path, audio_data, sample_rate)
    
#     # Добавляем запись в метаданные
#     metadata.append({
#         "audio_filename": audio_filename,
#         "transcription": example['text']  # Предполагаем, что текст находится в поле 'text'
#     })

# # Сохраняем метаданные в JSON файл
# with open('kyrgyz_dataset/metadata.json', 'w', encoding='utf-8') as f:
#     json.dump(metadata, f, ensure_ascii=False, indent=2)

# print("Обработка завершена!")
# print(f"Создано {len(metadata)} аудио файлов")
# print(f"Метаданные сохранены в kyrgyz_dataset/metadata.json")






























# from datasets import load_dataset, concatenate_datasets, Dataset, Audio, DatasetDict
# from scipy.io.wavfile import write
# import os
# import soundfile as sf
# import numpy as np
# from tqdm import tqdm
# import json
# import shutil

# # Устанавливаем путь для кэширования датасета
# CACHE_DIR = "dataset_cache"
# os.makedirs(CACHE_DIR, exist_ok=True)

# # Загружаем только первые 100 примеров из датасета (уменьшили количество)
# try:
#     ds = load_dataset(
#         "Simonlob/Kany_dataset_mk4_Base", 
#         split='train[:4000]',  # Уменьшили количество примеров
#         cache_dir=CACHE_DIR,  # Указываем отдельную директорию для кэша
#         download_mode="force_redownload"  # Принудительно перезагружаем
#     )
    
#     print(f"Размер датасета: {len(ds)} примеров")
#     print(f"Структура датасета: {ds[0]}")

#     # Создаем директорию для аудио файлов
#     os.makedirs('kyrgyz_dataset/audio_files', exist_ok=True)

#     metadata = []

#     for idx, example in enumerate(tqdm(ds, desc="Обработка датасета")):
#         try:
#             audio_data = example['audio']['array']
#             sample_rate = example['audio']['sampling_rate']

#             audio_filename = f"audio_{idx}.wav"
#             audio_path = os.path.join('kyrgyz_dataset/audio_files', audio_filename)

#             sf.write(audio_path, audio_data, sample_rate)

#             metadata.append({
#                 "audio_filename": audio_filename,
#                 "transcription": example['transcription']
#             })
#         except Exception as e:
#             print(f"Ошибка при обработке примера {idx}: {str(e)}")
#             continue

#     # Сохраняем метаданные
#     with open('kyrgyz_dataset/metadata.json', 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, ensure_ascii=False, indent=2)

#     print("Обработка завершена!")
#     print(f"Создано {len(metadata)} аудио файлов")
#     print(f"Метаданные сохранены в kyrgyz_dataset/metadata.json")

# except Exception as e:
#     print(f"Произошла ошибка: {str(e)}")
# finally:
#     # Очищаем кэш после использования
#     try:
#         shutil.rmtree(CACHE_DIR)
#         print("Кэш датасета очищен")
#     except Exception as e:
#         print(f"Ошибка при очистке кэша: {str(e)}")




































import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm
from num2words import num2words

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio, load_dataset
from kokoro.kokoro.model import KModel
from kokoro.kokoro.pipeline import KPipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Пути и параметры
DATASET_PATH = "kyrgyz_dataset"  # Путь к подготовленному датасету
OUTPUT_DIR = "whisper-kyrgyz"  # Путь для сохранения модели Whisper
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 4
MAX_INPUT_LENGTH_SECONDS = 30.0

def prepare_dataset(dataset_path: str, max_samples: int = 5) -> Dataset:
    """
    Подготовка датасета для обучения.
    
    Args:
        dataset_path: Путь к датасету
        max_samples: Максимальное количество примеров для обработки (по умолчанию 5)
    """
    try:
        logger.info(f"Загрузка датасета из {dataset_path}")
        metadata_path = os.path.join(dataset_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Ограничиваем количество примеров
        metadata = metadata[:max_samples]
        
        # Создаем список записей для датасета
        dataset_records = []
        audio_dir = os.path.join(dataset_path, "audio_files")
        
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Директория с аудио файлами не найдена: {audio_dir}")
        
        for item in tqdm(metadata, desc="Подготовка датасета"):
            audio_path = os.path.join(audio_dir, item["audio_filename"])
            if os.path.exists(audio_path):
                dataset_records.append({
                    "audio_path": audio_path,
                    "text": item["transcription"]
                })
            else:
                logger.warning(f"Аудио файл не найден: {audio_path}")
        
        if not dataset_records:
            raise ValueError("Нет доступных аудио файлов для обучения")
        
        # Создаем датасет
        dataset = Dataset.from_list(dataset_records)
        # Загружаем аудио
        dataset = dataset.cast_column("audio_path", Audio())
        
        logger.info(f"Датасет успешно загружен: {len(dataset)} примеров")
        return dataset
    
    except Exception as e:
        logger.error(f"Ошибка при подготовке датасета: {str(e)}")
        raise

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
                # Логируем длину текста
                logger.info(f"Длина текста: {len(text)} символов")
                
                # Ресемплинг аудио до 16000 Гц, если необходимо
                if audio_path["sampling_rate"] != 16000:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=audio_path["sampling_rate"],
                        new_freq=16000
                    )
                    audio_array = torch.from_numpy(audio_path["array"]).to(torch.float32)
                    if len(audio_array.shape) > 1:
                        audio_array = torch.mean(audio_array, dim=0)
                    audio_array = resampler(audio_array)
                    audio_path["array"] = audio_array.numpy().astype(np.float16)
                    audio_path["sampling_rate"] = 16000
                
                # Ограничиваем длину аудио до 3 секунд
                max_samples = int(3 * audio_path["sampling_rate"])
                audio_array = audio_path["array"]
                if len(audio_array) > max_samples:
                    audio_array = audio_array[:max_samples]
                
                # Логируем длину аудио
                logger.info(f"Длина аудио: {len(audio_array)} сэмплов")
                
                # Извлечение признаков аудио
                input_features = processor(
                    audio=audio_array,
                    sampling_rate=audio_path["sampling_rate"],
                    return_tensors="pt"
                ).input_features
                
                # Токенизируем текст с ограничением длины
                tokenized_text = processor.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=448,
                    padding="max_length"
                )
                
                # Создаем labels из токенизированного текста
                labels = tokenized_text["input_ids"].squeeze()
                
                # Проверяем размерности
                logger.info(f"Размерность input_features: {input_features.shape}")
                logger.info(f"Размерность labels: {labels.shape}")
                
                # Добавляем в батч
                processed_batch["input_features"].append(input_features.squeeze().numpy().astype(np.float16))
                processed_batch["labels"].append(labels.numpy().astype(np.int16))
                
                # Очистка памяти
                del audio_array
                del input_features
                del labels
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return processed_batch
        except Exception as e:
            logger.error(f"Ошибка при обработке примера: {str(e)}")
            return None
    
    # Создаем временную директорию для сохранения обработанных данных
    temp_dir = os.path.join(OUTPUT_DIR, "temp_processed")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Обрабатываем датасет частями
    processed_dataset = dataset.map(
        prepare_dataset,
        remove_columns=["audio_path", "text"],
        desc="Обработка датасета",
        batched=True,
        batch_size=1,  # Уменьшаем размер батча до 1
        writer_batch_size=1,  # Уменьшаем размер батча для записи до 1
        cache_file_name=os.path.join(temp_dir, "processed_dataset.arrow"),
        load_from_cache_file=True
    )
    
    # Проверяем размерность labels в обработанном датасете
    for example in processed_dataset:
        if len(example["labels"]) > 448:
            logger.warning(f"Найдены labels длиной {len(example['labels'])}. Обрезаем до 448.")
            example["labels"] = example["labels"][:448]
    
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
        
        # Проверяем размерность labels в обработанном датасете
        for example in processed_dataset:
            if len(example["labels"]) > 448:
                logger.warning(f"Найдены labels длиной {len(example['labels'])}. Обрезаем до 448.")
                example["labels"] = example["labels"][:448]
        
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

def synthesize_speech(text: str, output_path: str):
    """
    Синтез речи с использованием Kokoro.
    """
    try:
        logger.info("Инициализация Kokoro для синтеза речи...")
        # Загружаем модель Kokoro
        model = KModel()
        pipeline = KPipeline(model)
        
        # Синтезируем речь
        logger.info(f"Синтез речи для текста: {text}")
        audio = pipeline(text)
        
        # Сохраняем аудио
        torchaudio.save(output_path, audio.unsqueeze(0), 22050)
        logger.info(f"Аудио сохранено в {output_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при синтезе речи: {str(e)}")
        raise

def test_pipeline(model_path: str, audio_path: str, output_path: str):
    """
    Тестирование полного пайплайна: распознавание речи -> синтез речи.
    """
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
        
        # Распознавание речи с помощью Whisper
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        
        # Загрузка аудио
        audio, sample_rate = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = resampler(audio)
            sample_rate = 16000
        
        # Обработка аудио
        input_features = processor(
            audio=audio.squeeze().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features
        
        # Установка кыргызского языка
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ky", task="transcribe")
        
        # Предсказание
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        logger.info(f"Распознанный текст: {transcription}")
        
        # Синтез речи с помощью Kokoro
        synthesize_speech(transcription, output_path)
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании пайплайна: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Создание выходной директории
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Подготовка датасета (теперь с ограничением в 5 примеров)
        dataset = prepare_dataset(DATASET_PATH, max_samples=5)
        
        # Дообучение модели
        fine_tune_whisper(dataset, OUTPUT_DIR)
        
        logger.info("Обучение завершено успешно!")
        
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        raise



















