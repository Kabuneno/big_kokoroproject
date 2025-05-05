import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
import soundfile as sf
import numpy as np
from pathlib import Path
import os
from loguru import logger

class Config:
    MODEL_PATH = "output"  # Путь к локальной модели
    OUTPUT_DIR = Path("output")
    SAMPLE_RATE = 16000
    MAX_LENGTH = 256
    LANGUAGE = "ky"  # Код языка для кыргызского

def load_model():
    """Загрузка модели и токенизатора."""
    try:
        logger.info("Загрузка локальной модели...")
        
        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
        logger.info("Токенизатор загружен успешно")
        
        # Загрузка модели
        model = AutoModelForSpeechSeq2Seq.from_pretrained(Config.MODEL_PATH)
        logger.info("Модель загружена успешно")
        
        # Загрузка процессора Whisper для обработки аудио
        processor = WhisperProcessor.from_pretrained(Config.MODEL_PATH)
        logger.info("Процессор загружен успешно")
        
        # Переводим модель в режим оценки
        model.eval()
        
        return tokenizer, model, processor
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise

def generate_speech(text, tokenizer, model, processor):
    """Генерация речи из текста."""
    try:
        logger.info(f"Начало генерации речи для текста: {text}")
        
        # Проверка входного текста
        if not text or not isinstance(text, str):
            raise ValueError("Текст должен быть непустой строкой")
        
        # Подготовка входных данных
        logger.info("Подготовка входных данных...")
        
        # Создаем фиктивное аудио для генерации
        dummy_audio = np.zeros((Config.SAMPLE_RATE,), dtype=np.float32)
        
        # Обрабатываем аудио через процессор
        input_features = processor(
            dummy_audio,
            sampling_rate=Config.SAMPLE_RATE,
            return_tensors="pt"
        ).input_features
        
        # Подготавливаем текст для модели
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=Config.LANGUAGE,
            task="transcribe"
        )
        
        # Генерация речи
        logger.info("Генерация речи...")
        with torch.no_grad():
            outputs = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=Config.MAX_LENGTH,
                num_beams=5,
                temperature=0.7,
                return_dict_in_generate=True
            )
        
        # Извлечение аудио
        logger.info("Извлечение аудио...")
        audio = outputs.audio_values.squeeze().cpu().numpy()
        
        # Проверка формы аудио
        if len(audio.shape) != 1:
            raise ValueError(f"Неожиданная форма аудио: {audio.shape}")
        
        # Сохранение аудио
        output_path = "generated_speech.wav"
        logger.info(f"Сохранение аудио в {output_path}...")
        sf.write(output_path, audio, Config.SAMPLE_RATE)
        
        logger.info("Генерация речи успешно завершена")
        return output_path
    
    except Exception as e:
        logger.error(f"Ошибка при генерации речи: {str(e)}")
        return None

def main():
    """Основная функция."""
    try:
        # Загрузка модели
        tokenizer, model, processor = load_model()
        
        while True:
            print("\nВведите текст на кыргызском (или 'выход' для завершения):")
            text = input().strip()
            
            if not text:
                print("Пожалуйста, введите текст")
                continue
                
            if text.lower() == 'выход':
                print("Выход...")
                break
            
            # Генерация речи
            audio_path = generate_speech(text, tokenizer, model, processor)
            
            if audio_path:
                print(f"\nРечь сгенерирована и сохранена в {audio_path}")
                print("Воспроизведение...")
                # Воспроизведение аудио
                os.system(f"start {audio_path}")
            else:
                print("Не удалось сгенерировать речь. Проверьте логи для подробностей.")
    
    except Exception as e:
        logger.error(f"Ошибка в main: {str(e)}")
        raise

if __name__ == "__main__":
    main()