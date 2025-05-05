import os
import json
import torch
import numpy as np
import torchaudio
import random
from datasets import Dataset, Audio
from transformers import (
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    Seq2SeqTrainingArguments
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import logging
import shutil
import requests
from pathlib import Path
import subprocess
import wandb

# Определение преобразования в мел-спектрограмму
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    n_mels=80,
    f_min=80,
    f_max=7600,
    window_fn=torch.hann_window,
    normalized=True
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Настройки
model_id = "microsoft/speecht5_tts"
vocoder_id = "microsoft/speecht5_hifigan"
output_dir = "./kyrgyz-tts-model"
dataset_dir = "./kyrgyz_dataset"
audio_dir = os.path.join(dataset_dir, "audio_files")
metadata_path = os.path.join(dataset_dir, "metadata.json")
kokoro_dir = "./kokoro"  # Путь к локальной установке Kokoro
kokoro_tts_dir = os.path.join(kokoro_dir, "tts")  # Папка для TTS модулей в Kokoro
kyrgyz_chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяңөүһ"

# Настройки для обучения
max_train_samples = 1000
eval_samples = 50
batch_size = 4
learning_rate = 1e-5  # Уменьшаем learning rate
num_train_epochs = 30
sampling_rate = 16000
warmup_steps = 500  # Добавляем warmup steps

# Initialize wandb
# def init_wandb():
#     wandb.init(
#         project="kyrgyz_tts",
#         name="kyrgyz_tts_training",
#         config={
#             "max_train_samples": max_train_samples,
#             "eval_samples": eval_samples,
#             "batch_size": batch_size,
#             "learning_rate": learning_rate,
#             "num_train_epochs": num_train_epochs,
#             "sampling_rate": sampling_rate,
#             "warmup_steps": warmup_steps
#         }
#     )
#     return wandb.run
# else:
#     wandb_run = None

# Создаем директории
os.makedirs(output_dir, exist_ok=True)
os.makedirs("./model_cache", exist_ok=True)

def load_kyrgyz_dataset(dataset_dir, metadata_path, audio_dir):
    """
    Загружает кыргызский датасет из существующей структуры
    """
    logger.info(f"Загрузка датасета из {dataset_dir}")
    
    # Проверяем существование директорий и файлов
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Директория датасета не найдена: {dataset_dir}")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Директория с аудио не найдена: {audio_dir}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
    
    # Загружаем метаданные
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if not metadata:
        raise ValueError("Метаданные пусты")
    
    # Создаем словарь для датасета
    dataset_dict = {
        "audio": [],
        "text": [],
        "speaker_id": []
    }
    
    # Добавляем записи из метаданных
    for item in metadata:
        audio_path = os.path.join(audio_dir, item["audio_filename"])
        if not os.path.exists(audio_path):
            logger.warning(f"Аудиофайл не найден: {audio_path}")
            continue
        
        dataset_dict["audio"].append(audio_path)
        dataset_dict["text"].append(item["transcription"])
        dataset_dict["speaker_id"].append("kyrgyz_speaker")  # Используем один идентификатор диктора
    
    if not dataset_dict["audio"]:
        raise ValueError("Не найдено ни одного аудиофайла")
    
    logger.info(f"Загружено {len(dataset_dict['audio'])} примеров")
    
    # Создаем датасет
    dataset = Dataset.from_dict(dataset_dict)
    
    # Добавляем обработчик аудио
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    
    # Разделяем на тренировочную и тестовую выборки
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    
    return train_test_split

def setup_kyrgyz_tokenizer(processor):
    """
    Настраивает токенизатор для поддержки кыргызского языка
    """
    logger.info("Настройка токенизатора для кыргызского языка")
    
    # Получаем базовый токенизатор
    tokenizer = processor.tokenizer
    
    # Добавляем кыргызские символы в словарь
    new_tokens = []
    for char in kyrgyz_chars:
        if tokenizer.convert_tokens_to_ids(char) == tokenizer.unk_token_id:
            new_tokens.append(char)
    
    # Добавляем новые токены
    if new_tokens:
        num_added = tokenizer.add_tokens(new_tokens)
        logger.info(f"Добавлено {num_added} новых токенов в словарь")
    
    return tokenizer

def extract_speaker_embeddings(speaker_wavs, processor, model):
    """
    Extracts speaker embeddings from audio files using a simple approach
    """
    logger.info("Extracting speaker embeddings from audio samples")
    
    try:
        # Process a few audio files to get representative audio
        embedding_dim = 512  # Standard embedding dimension for SpeechT5
        embedding_sum = torch.zeros(embedding_dim, dtype=torch.float32)
        count = 0
        
        for wav_file in speaker_wavs[:10]:  # Use first 10 files
            # Load the audio
            waveform, sample_rate = torchaudio.load(wav_file)
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Simple feature extraction using audio statistics
            waveform = waveform.squeeze(0)  # Remove channel dimension
            
            # Create a feature vector using audio statistics
            if len(waveform) > 0:
                # Create a deterministic embedding from audio statistics
                audio_features = torch.tensor([
                    waveform.mean().item(),
                    waveform.std().item(),
                    waveform.max().item(),
                    waveform.min().item(),
                    waveform.median().item()
                ], dtype=torch.float32)
                
                # Expand to embedding dimension
                torch.manual_seed(42 + count)  # Deterministic but different for each file
                random_vectors = torch.randn(5, embedding_dim // 5, dtype=torch.float32)
                expanded_features = audio_features.unsqueeze(-1) * random_vectors
                expanded_features = expanded_features.reshape(-1)
                
                # Pad or truncate to embedding dimension
                if expanded_features.size(0) < embedding_dim:
                    padding = torch.zeros(embedding_dim - expanded_features.size(0), dtype=torch.float32)
                    expanded_features = torch.cat([expanded_features, padding])
                else:
                    expanded_features = expanded_features[:embedding_dim]
                
                embedding_sum += expanded_features
                count += 1
        
        if count > 0:
            speaker_embedding = embedding_sum / count
        else:
            # Fallback if no files processed
            torch.manual_seed(42)
            speaker_embedding = torch.randn(embedding_dim, dtype=torch.float32)
        
        # Normalize the embedding
        speaker_embedding = speaker_embedding / (torch.norm(speaker_embedding) + 1e-8)
        
        logger.info(f"Created speaker embedding with shape {speaker_embedding.shape}")
        return speaker_embedding
        
    except Exception as e:
        logger.error(f"Error extracting speaker embeddings: {e}")
        logger.warning("Falling back to synthetic embeddings")
        
        # Fallback to synthetic embeddings
        embedding_dim = 512
        torch.manual_seed(42)
        speaker_embedding = torch.randn(embedding_dim, dtype=torch.float32)
        speaker_embedding = speaker_embedding / (torch.norm(speaker_embedding) + 1e-8)
        
        return speaker_embedding

class TTSDataCollator:
    def __init__(self, processor: Any, speaker_embeddings: Dict[str, torch.Tensor]):
        self.processor = processor
        self.speaker_embeddings = speaker_embeddings
    
    def __call__(self, features: List[Dict[str, Union[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        labels = []
        speaker_embeddings = []

        for feature in features:
            # Токенизация текста
            tokenized = self.processor(
                text=feature["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=200,
                truncation=True
            )
            
            input_ids.append(tokenized.input_ids.squeeze())
            attention_mask.append(tokenized.attention_mask.squeeze())
            
            # Обработка аудио
            audio = feature["audio"]
            # Преобразование аудио в мел-спектрограмму
            waveform = torch.tensor(audio).unsqueeze(0)  # [1, N]
            mel_spec = mel_transform(waveform).to(torch.float32)  # Преобразуем в FloatTensor
            mel_spec = mel_spec.squeeze(0).transpose(0, 1)  # [T, 80]
            
            # Убедитесь, что mel_spec является тензором
            if isinstance(mel_spec, torch.Tensor):
                labels.append(mel_spec)  # Используем мел-спектрограмму как labels
                # Добавьте эмбеддинг диктора
                speaker_id = feature["speaker_id"]
                if speaker_id in self.speaker_embeddings:
                    speaker_embeddings.append(self.speaker_embeddings[speaker_id])
                else:
                    logger.warning(f"Эмбеддинг для диктора {speaker_id} не найден.")
            else:
                logger.warning("mel_spec не является тензором, пропускаем этот пример.")
                continue  # Пропускаем этот пример, если mel_spec не является тензором
        
        fixed_length = 400  # Установите желаемую длину
        # Паддинг меток
        if labels:  # Проверяем, что labels не пустой
            # max_length = max(label.size(0) for label in labels)
            padded_labels = []
            for label in labels:
                if label.size(0) < fixed_length:
                    # Pad with zeros
                    padding = torch.zeros(fixed_length - label.size(0), label.size(1))
                    padded_label = torch.cat((label, padding), dim=0)
                elif label.size(0) > fixed_length:
                    padded_label = label[:fixed_length]
                else:
                    padded_label = label
                padded_labels.append(padded_label)

            # Проверка на пустоту speaker_embeddings
            if speaker_embeddings:
                return {
                    "input_ids": torch.stack(input_ids).to(torch.long),
                    "attention_mask": torch.stack(attention_mask).to(torch.long),
                    "labels": torch.stack(padded_labels).to(torch.float32),
                    "speaker_embeddings": torch.stack(speaker_embeddings).to(torch.float32)
                }
            else:
                logger.error("Не найдено ни одного валидного эмбеддинга диктора.")
                return None  # Или обработайте это другим образом
        else:
            logger.error("Не найдено ни одной валидной метки.")
            return None  # Или обработайте это другим образом

def prepare_dataset(example, processor):
    try:
        # Получение аудио
        audio = example["audio"]["array"] if isinstance(example["audio"], dict) else example["audio"]
        audio = audio.astype(np.float32)

        # Нормализация аудио
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio) + 1e-8)

        waveform = torch.tensor(audio).unsqueeze(0)  # [1, N]

        # Преобразуем в мел-спектрограмму
        mel_spec = mel_transform(waveform).to(torch.float32)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)  # [T, 80]

        # Нормализация мел-спектрограммы
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        # Установите фиксированную длину для мел-спектрограмм
        fixed_length = 400
        if mel_spec.size(0) > fixed_length:
            # Берем случайный отрезок для разнообразия
            start = random.randint(0, mel_spec.size(0) - fixed_length)
            mel_spec = mel_spec[start:start + fixed_length, :]
        elif mel_spec.size(0) < fixed_length:
            padding = torch.zeros(fixed_length - mel_spec.size(0), 80)
            mel_spec = torch.cat((mel_spec, padding), dim=0)

        # Ограничиваем длину аудио до 3 секунд
        max_samples = sampling_rate * 3
        if len(audio) > max_samples:
            # Берем случайный отрезок для разнообразия
            start = random.randint(0, len(audio) - max_samples)
            audio = audio[start:start + max_samples]

        # Токенизация текста
        inputs = processor(
            text=example["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=200,
            truncation=True
        )

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "audio": audio,
            "text": example["text"],
            "speaker_id": example["speaker_id"],
            "labels": mel_spec
        }
    except Exception as e:
        logger.error(f"Ошибка при подготовке датасета: {str(e)}")
        return None

def train_tts_model():
    """
    Основная функция обучения TTS-модели на кыргызском языке
    """
    logger.info("Начинаем обучение TTS-модели на кыргызском языке")

    # Загружаем датасет
    dataset = load_kyrgyz_dataset(dataset_dir, metadata_path, audio_dir)

    # Загружаем модель и процессор
    logger.info(f"Загрузка модели {model_id}")
    try:
        model = SpeechT5ForTextToSpeech.from_pretrained(model_id, ignore_mismatched_sizes=True).to(torch.float32)
        processor = SpeechT5Processor.from_pretrained(model_id)
        vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id)
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise

    # Настраиваем токенизатор
    tokenizer = setup_kyrgyz_tokenizer(processor)
    model.resize_token_embeddings(len(tokenizer))

    # Эмбеддинги диктора
    all_audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
    speaker_wavs = random.sample(all_audio_files, min(20, len(all_audio_files)))
    speaker_embedding = extract_speaker_embeddings(speaker_wavs, processor, model)
    speaker_embeddings = {"kyrgyz_speaker": speaker_embedding}

    # Обрезаем датасет если слишком большой
    if len(dataset["train"]) > max_train_samples:
        dataset["train"] = dataset["train"].select(range(max_train_samples))
    if len(dataset["test"]) > eval_samples:
        dataset["test"] = dataset["test"].select(range(eval_samples))

    # Подготовка датасета
    logger.info("Подготовка датасета")
    prepared_dataset = {}
    for split in dataset:
        mapped_dataset = dataset[split].map(
            lambda x: prepare_dataset(x, processor),
            remove_columns=["audio"] if "audio" in dataset[split].column_names else [],
            batched=False,
            num_proc=1
        )
        prepared_dataset[split] = mapped_dataset.filter(
            lambda x: x is not None and all(k in x for k in ["input_ids", "attention_mask", "audio", "text"])
        )

    # Пример данных
    sample = prepared_dataset["train"][0]
    logger.info(f"Пример данных: текст={sample['text']}, аудио длина={len(sample['audio'])}, speaker_id={sample['speaker_id']}")

    # Коллатор
    data_collator = TTSDataCollator(
        processor=processor,
        speaker_embeddings=speaker_embeddings
    )

    # Проверка коллатора
    test_batch = data_collator([prepared_dataset["train"][0], prepared_dataset["train"][1]])
    logger.info(f"Размерности батча: input_ids={test_batch['input_ids'].shape}, labels={test_batch['labels'].shape}")

    # Инициализация wandb
    try:
        wandb_run = wandb.init(
            project="kokoro-kyrgyz-tts",
            name="speecht5-kyrgyz-training",
            config={
                "model_id": model_id,
                "vocoder_id": vocoder_id,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_train_epochs": num_train_epochs,
                "sampling_rate": sampling_rate,
                "max_train_samples": max_train_samples,
                "eval_samples": eval_samples,
                "output_dir": output_dir
            }
        )
    except Exception as e:
        logger.warning(f"Не удалось инициализировать wandb: {e}")
        wandb_run = None

    # Параметры обучения
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,  # Добавляем warmup steps
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Используем правильную метрику
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=["wandb"] if wandb_run else "none",
        run_name=wandb_run.name if wandb_run else None
    )

    from transformers import EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

    # Тренер (создаем ТОЛЬКО ПОСЛЕ всего выше)
    from transformers import Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset["test"],
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    # Обучение
    logger.info("Запуск обучения модели")
    trainer.train()
    


    # Перед сохранением модели в train_tts_model()
    logger.info(f"Размер словаря токенизатора: {len(processor.tokenizer)}")
    logger.info("Пример токенизации:")
    sample_text = "Саламатсызбы"
    sample_tokens = processor(text=sample_text, return_tensors="pt")
    logger.info(f"Текст: {sample_text} -> Токены: {sample_tokens['input_ids']}")

# Сохраняем tokenizer отдельно для проверки
    processor.tokenizer.save_pretrained(output_dir)
    # Сохраняем всё
    logger.info(f"Сохраняем модель в {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    torch.save(speaker_embeddings, os.path.join(output_dir, "speaker_embeddings.pt"))
    vocoder.save_pretrained(os.path.join(output_dir, "vocoder"))

    return model, processor, vocoder, speaker_embeddings





def synthesize_kyrgyz_speech(text, model, processor, vocoder, speaker_embeddings, output_path=None):
    """
    Синтезирует речь на кыргызском языке
    """
    logger.info(f"Синтез речи: '{text}'")
    
    # Токенизируем текст
    inputs = processor(text=text, return_tensors="pt")
    
    # Получаем эмбеддинг диктора
    speaker_embedding = speaker_embeddings["kyrgyz_speaker"].unsqueeze(0)
    
    # Генерируем спектрограмму
    speech = model.generate_speech(
        inputs["input_ids"], 
        speaker_embeddings=speaker_embedding,
        vocoder=vocoder
    )
    
    # Сохраняем аудио если указан путь
    if output_path:
        torchaudio.save(
            output_path,
            speech.unsqueeze(0),
            sample_rate=sampling_rate
        )
        logger.info(f"Аудио сохранено в {output_path}")
    
    return speech

def integrate_with_kokoro():
    """
    Интегрирует обученную модель с Kokoro
    """
    logger.info("Начинаем интеграцию с Kokoro")
    
    # Проверяем наличие директории Kokoro
    if not os.path.exists(kokoro_dir):
        logger.error(f"Директория Kokoro не найдена: {kokoro_dir}")
        raise FileNotFoundError(f"Директория Kokoro не найдена: {kokoro_dir}")
    
    # Создаем директорию для кыргызской TTS в Kokoro
    kokoro_kyrgyz_tts_dir = os.path.join(kokoro_tts_dir, "kyrgyz_tts")
    os.makedirs(kokoro_kyrgyz_tts_dir, exist_ok=True)
    
    # Копируем модель в директорию Kokoro
    logger.info(f"Копирование модели в {kokoro_kyrgyz_tts_dir}")
    
    # Копируем основную модель
    shutil.copytree(output_dir, kokoro_kyrgyz_tts_dir, dirs_exist_ok=True)
    
    # Создаем интерфейсный скрипт для Kokoro
    interface_script = os.path.join(kokoro_kyrgyz_tts_dir, "kyrgyz_tts_interface.py")
    
    interface_content = '''
import os
import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

class KyrgyzTTS:
    def __init__(self):
        self.model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_dir, ignore_mismatched_sizes=True)
        self.processor = SpeechT5Processor.from_pretrained(self.model_dir)
        self.vocoder = SpeechT5HifiGan.from_pretrained(os.path.join(self.model_dir, "vocoder"))
        self.speaker_embeddings = torch.load(os.path.join(self.model_dir, "speaker_embeddings.pt"), weights_only=True)
        self.sampling_rate = 16000
        
    def synthesize(self, text, output_path=None):
        """
        Синтезирует кыргызскую речь
        """
        # Токенизируем текст
        inputs = self.processor(text=text, return_tensors="pt")
        
        # Получаем эмбеддинг диктора
        speaker_embedding = self.speaker_embeddings["kyrgyz_speaker"].unsqueeze(0)
        
        # Генерируем спектрограмму
        speech = self.model.generate_speech(
            inputs["input_ids"], 
            speaker_embeddings=speaker_embedding,
            vocoder=self.vocoder
        )
        
        # Сохраняем аудио если указан путь
        if output_path:
            torchaudio.save(
                output_path,
                speech.unsqueeze(0),
                sample_rate=self.sampling_rate
            )
        
        return speech.numpy(), self.sampling_rate
        
# Для тестирования
if __name__ == "__main__":
    tts = KyrgyzTTS()
    speech, sr = tts.synthesize("Саламатсызбы, мен Кокоромун!", "test_output.wav")
    print(f"Синтез выполнен успешно. Аудио сохранено в test_output.wav")
'''
    
    # Записываем интерфейсный скрипт
    with open(interface_script, 'w', encoding='utf-8') as f:
        f.write(interface_content)
    
    # Создаем файл инициализации для импорта в Kokoro
    init_file = os.path.join(kokoro_kyrgyz_tts_dir, "__init__.py")
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write("from .kyrgyz_tts_interface import KyrgyzTTS\n\n__all__ = ['KyrgyzTTS']")
    
    # Создаем конфигурационный файл для Kokoro
    config_file = os.path.join(kokoro_kyrgyz_tts_dir, "config.json")
    config_content = '''{
  "name": "kyrgyz_tts",
  "version": "1.0.0",
  "description": "Кыргызский модуль синтеза речи для Kokoro",
  "language": "ky",
  "author": "User",
  "dependencies": ["transformers>=4.27.0", "torchaudio>=0.13.0"]
}'''
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    # Создаем примеры интеграции для Kokoro
    example_file = os.path.join(kokoro_kyrgyz_tts_dir, "example_usage.py")
    example_content = '''
# Пример интеграции кыргызского TTS с Kokoro

# Импортируем модуль Kokoro (замените на ваш путь импорта)
import sys
sys.path.append("../../")  # Адаптируйте путь к корневой директории Kokoro

try:
    # Импортируем основной класс Kokoro
    from kokoro import Kokoro
    
    # Импортируем модуль кыргызского TTS
    from kyrgyz_tts.kyrgyz_tts_interface import KyrgyzTTS
    
    # Создаем экземпляр Kokoro
    kokoro = Kokoro()
    
    # Регистрируем кыргызский TTS в Kokoro
    kokoro.register_tts("kyrgyz", KyrgyzTTS())
    
    # Пример генерации кыргызской речи
    kyrgyz_text = "Саламатсызбы! Мен Кокоро, мен кыргызча сүйлөйм."
    kokoro.speak(kyrgyz_text, language="kyrgyz")
    
    print("Кыргызский TTS успешно интегрирован с Kokoro!")
    
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что Kokoro правильно установлен и доступен.")
except Exception as e:
    print(f"Ошибка при интеграции: {e}")
'''
    
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(example_content)
    
    logger.info("Интеграция с Kokoro завершена!")
    return kokoro_kyrgyz_tts_dir

def test_kokoro_integration(kokoro_kyrgyz_tts_dir):
    """
    Тестирует интеграцию с Kokoro
    """
    logger.info("Тестирование интеграции с Kokoro")
    
    # Тестируем модуль TTS
    test_script = os.path.join(kokoro_kyrgyz_tts_dir, "kyrgyz_tts_interface.py")
    
    try:
        subprocess.run(["python", test_script], check=True)
        logger.info("Тестирование прошло успешно!")
        
        # Открываем директорию с результатами
        if os.name == 'nt':  # Windows
            os.startfile(kokoro_kyrgyz_tts_dir)
        elif os.name == 'posix':  # Linux/Mac
            subprocess.run(["xdg-open", kokoro_kyrgyz_tts_dir], check=False)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при тестировании: {e}")

if __name__ == "__main__":
    try:
        # Проверяем наличие датасета
        if not os.path.exists(metadata_path) or not os.path.exists(audio_dir):
            logger.error("Датасет не найден. Убедитесь, что структура датасета соответствует требованиям.")
            logger.info(f"Ожидаемый путь к метаданным: {metadata_path}")
            logger.info(f"Ожидаемый путь к аудио: {audio_dir}")
        else:
            # Обучаем модель
            model, processor, vocoder, speaker_embeddings = train_tts_model()
            
            # Проверяем работу модели
            test_text = "Саламатсызбы, мен Кокоромун, кыргызча сүйлөйм!"
            output_path = os.path.join(output_dir, "test_synthesis.wav")
            
            synthesize_kyrgyz_speech(
                test_text,
                model,
                processor,
                vocoder,
                speaker_embeddings,
                output_path
            )
            
            # Интегрируем с Kokoro
            kokoro_module_dir = integrate_with_kokoro()
            
            # Тестируем интеграцию
            test_kokoro_integration(kokoro_module_dir)
            
            logger.info("""
            ========================================================
            Обучение и интеграция успешно завершены!
            
            Модель сохранена в: {}
            Тестовое аудио: {}
            Модуль для Kokoro: {}
            
            Для использования в Kokoro добавьте следующий код:
            
            from kyrgyz_tts.kyrgyz_tts_interface import KyrgyzTTS
            kokoro.register_tts("kyrgyz", KyrgyzTTS())
            kokoro.speak("Ваш текст на кыргызском", language="kyrgyz")
            ========================================================
            """.format(output_dir, output_path, kokoro_module_dir))
            
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}", exc_info=True)
