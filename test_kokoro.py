import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
from loguru import logger
from gtts import gTTS
import os

# Настройки
class Config:
    MODEL_DIR = Path("output")  # Директория с обученной моделью
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 30.0  # секунды

def load_model():
    """Загрузка обученной модели и процессора."""
    logger.info("Загрузка модели...")
    processor = WhisperProcessor.from_pretrained(Config.MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_DIR)
    return processor, model

def record_audio(duration=3.0):
    """Запись аудио с микрофона."""
    logger.info(f"Запись аудио в течение {duration} секунд...")
    audio = sd.rec(
        int(duration * Config.SAMPLE_RATE),
        samplerate=Config.SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio

def save_audio(audio, filename):
    """Сохранение аудио в файл."""
    sf.write(filename, audio, Config.SAMPLE_RATE)
    logger.info(f"Аудио сохранено в {filename}")

def transcribe_audio(audio_path, processor, model):
    """Транскрибация аудио в текст."""
    try:
        # Загрузка аудио
        waveform, sample_rate = torchaudio.load(audio_path)
        
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
        
        # Извлечение признаков
        input_features = processor(
            audio=waveform.squeeze().numpy(),
            sampling_rate=Config.SAMPLE_RATE,
            return_tensors="pt"
        ).input_features
        
        # Генерация текста
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return transcription[0]
    
    except Exception as e:
        logger.error(f"Ошибка при транскрибации: {str(e)}")
        return None

def generate_speech(text, processor, model):
    """Генерация речи из текста с помощью обученной модели."""
    try:
        # Токенизация текста
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=448
        )
        
        # Генерация аудио
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1000,
                num_beams=5,
                temperature=0.7,
                return_dict_in_generate=True
            )
        
        # Извлечение аудио из выходных данных
        if hasattr(outputs, 'audio_values'):
            audio = outputs.audio_values.squeeze().cpu().numpy()
        else:
            # Если модель не возвращает аудио напрямую, попробуем декодировать
            audio = processor.decode(outputs.sequences[0], skip_special_tokens=True)
            # Преобразуем текст обратно в аудио
            audio = np.array([ord(c) for c in audio], dtype=np.float32)
            audio = (audio - audio.min()) / (audio.max() - audio.min()) * 2 - 1  # Нормализация
        
        # Сохранение аудио
        output_path = "generated_speech.wav"
        sf.write(output_path, audio, Config.SAMPLE_RATE)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Ошибка при генерации речи: {str(e)}")
        return None

def main():
    """Основная функция для тестирования модели."""
    try:
        # Загрузка модели
        processor, model = load_model()
        
        while True:
            print("\nВыберите режим работы:")
            print("1. Распознавание речи (ASR)")
            print("2. Генерация речи (TTS)")
            print("3. Выход")
            
            choice = input("Ваш выбор (1-3): ")
            
            if choice == "1":
                # Режим распознавания речи
                print("\nГоворите...")
                audio = record_audio()
                save_audio(audio, "temp.wav")
                
                transcription = transcribe_audio("temp.wav", processor, model)
                if transcription:
                    print(f"\nРаспознанный текст: {transcription}")
            
            elif choice == "2":
                # Режим генерации речи
                text = input("\nВведите текст на кыргызском: ")
                audio_path = generate_speech(text, processor, model)
                
                if audio_path:
                    print(f"\nРечь сгенерирована и сохранена в {audio_path}")
                    print("Воспроизведение...")
                    # Воспроизведение аудио
                    audio, sr = sf.read(audio_path)
                    sd.play(audio, sr)
                    sd.wait()
            
            elif choice == "3":
                print("Выход...")
                break
            
            else:
                print("Неверный выбор. Попробуйте снова.")
    
    except Exception as e:
        logger.error(f"Ошибка в main: {str(e)}")
        raise

if __name__ == "__main__":
    main()

