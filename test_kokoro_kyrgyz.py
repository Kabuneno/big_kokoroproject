
# # Импортируем модуль
# # from kokoro.tts.kyrgyz_tts_interface import KyrgyzTTS
# from kokoro.tts.kyrgyz_tts.kyrgyz_tts_interface import KyrgyzTTS
# import kokoro
# # Регистрируем в Kokoro
# kokoro.register_tts("kyrgyz", KyrgyzTTS())

# # Синтезируем речь
# kokoro.speak("Мен кыргызча сүйлөйм", language="kyrgyz")



# from kokoro.tts.kyrgyz_tts.kyrgyz_tts_interface import KyrgyzTTS

# # # Инициализация модели
# # tts_model = KyrgyzTTS()


# # # Синтезируем речь
# # text = "Мен кыргызча сүйлөйм"
# # audio_output = tts_model.speak(text, language="kyrgyz")
# # tts_model.check_tokenizer()
# # # Сохранение аудио в файл (если метод возвращает аудио)
# # with open("output.wav", "wb") as audio_file:
# #     audio_file.write(audio_output)


# # tts_model = KyrgyzTTS()

# # # Проверка токенизатора
# # tts_model.check_tokenizer()

# # # Тестовые тексты
# # test_texts = [
# #     "Саламатсызбы",
# #     "Мен кыргызча сүйлөйм",
# #     "Кыргызстан"
# # ]

# # for text in test_texts:
# #     try:
# #         print(f"\nТестируем: '{text}'")
# #         audio, sr = tts_model.synthesize(text, f"output_{text[:10]}.wav")
# #         print(f"Успешно синтезировано: {len(audio)} samples")
# #     except Exception as e:
# #         print(f"Ошибка для текста '{text}': {str(e)}")


# tts = KyrgyzTTS()
# tts.check_tokenizer()  # Проверка токенизации
# audio, sr = tts.synthesize("Саламатсызбы", "output.wav")  # Синтез речи








import os
import torch
import torchaudio
import numpy as np
import logging
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model paths and settings
output_dir = "./kyrgyz-tts-model"
test_output_dir = "./test_results"
sampling_rate = 16000
os.makedirs(test_output_dir, exist_ok=True)

def load_model():
    """Load the trained Kyrgyz TTS model components"""
    logger.info("Loading model components...")
    
    try:
        model = SpeechT5ForTextToSpeech.from_pretrained(output_dir, ignore_mismatched_sizes=True)
        processor = SpeechT5Processor.from_pretrained(output_dir)
        vocoder = SpeechT5HifiGan.from_pretrained(os.path.join(output_dir, "vocoder"))
        speaker_embeddings = torch.load(os.path.join(output_dir, "speaker_embeddings.pt"))
        
        logger.info("Model components loaded successfully")
        return model, processor, vocoder, speaker_embeddings
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def plot_spectrogram(speech, title):
    """Plot the spectrogram of the generated speech"""
    plt.figure(figsize=(10, 4))
    plt.imshow(speech.T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    return plt

def synthesize_and_save(model, processor, vocoder, speaker_embeddings, text, output_path):
    """Synthesize speech and save to file"""
    # Tokenize text
    inputs = processor(text=text, return_tensors="pt")
    
    # Get speaker embedding
    speaker_embedding = speaker_embeddings["kyrgyz_speaker"].unsqueeze(0)
    
    # Generate spectrogram and convert to speech
    speech = model.generate_speech(
        inputs["input_ids"], 
        speaker_embeddings=speaker_embedding,
        vocoder=vocoder
    )
    
    # Save audio
    torchaudio.save(
        output_path,
        speech.unsqueeze(0),
        sample_rate=sampling_rate
    )
    
    return speech

def test_model_with_phrases():
    """Test the model with various Kyrgyz phrases"""
    logger.info("Starting model testing")
    
    # Load model components
    model, processor, vocoder, speaker_embeddings = load_model()
    
    # Test phrases - include a range of Kyrgyz phrases
    test_phrases = [
        "Саламатсызбы, мен Кокоромун!",
        "Кыргызстан - кооз өлкө.",
        "Бул кыргыз тилиндеги синтездөө тести.",
        "Жакшы жашоо - бул бактылуу жашоо.",
        # Add more phrases featuring different Kyrgyz phonetics
    ]
    
    # Prepare HTML report
    html_report = "<html><head><title>Kyrgyz TTS Test Results</title></head><body>"
    html_report += "<h1>Kyrgyz TTS Model Test Results</h1>"
    html_report += "<table border='1'><tr><th>Text</th><th>Audio</th></tr>"
    
    for i, phrase in enumerate(test_phrases):
        logger.info(f"Testing phrase {i+1}: {phrase}")
        
        # Path for audio output
        output_path = os.path.join(test_output_dir, f"test_kyrgyz_speech_{i+1}.wav")
        
        try:
            # Synthesize speech
            speech = synthesize_and_save(model, processor, vocoder, speaker_embeddings, phrase, output_path)
            
            # Add to report
            html_report += f"<tr><td>{phrase}</td><td><audio controls src='{output_path}'></audio></td></tr>"
            
            logger.info(f"Generated audio saved to {output_path}")
            
            # Optional: If in notebook environment, play the audio
            if 'google.colab' in str(get_ipython()):
                display(Audio(speech.numpy(), rate=sampling_rate))
        
        except Exception as e:
            logger.error(f"Error processing phrase: {e}")
            html_report += f"<tr><td>{phrase}</td><td>Error: {str(e)}</td></tr>"
    
    html_report += "</table></body></html>"
    
    # Save HTML report
    with open(os.path.join(test_output_dir, "test_report.html"), "w", encoding="utf-8") as f:
        f.write(html_report)
    
    logger.info(f"Testing complete. Results saved to {test_output_dir}")
    logger.info(f"HTML report saved to {os.path.join(test_output_dir, 'test_report.html')}")

def test_tokenizer():
    """Test the tokenizer to ensure it handles Kyrgyz characters correctly"""
    logger.info("Testing tokenizer for Kyrgyz language support")
    
    # Load processor
    processor = SpeechT5Processor.from_pretrained(output_dir)
    
    # Test characters
    kyrgyz_chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяңөүһ"
    special_chars = "ңөүһ"  # Kyrgyz-specific characters
    
    # Test all characters
    logger.info("Testing all Kyrgyz characters:")
    token_ids = processor(text=kyrgyz_chars, return_tensors="pt").input_ids
    logger.info(f"Input: {kyrgyz_chars}")
    logger.info(f"Token IDs: {token_ids}")
    
    # Test special Kyrgyz characters
    logger.info("\nTesting special Kyrgyz characters:")
    for char in special_chars:
        token_id = processor(text=char, return_tensors="pt").input_ids
        logger.info(f"Character: '{char}' -> Token ID: {token_id}")

def main():
    """Main test function"""
    try:
        logger.info("Starting Kyrgyz TTS model testing")
        
        # Test tokenizer first
        test_tokenizer()
        
        # Test model with phrases
        test_model_with_phrases()
        
        logger.info("Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Testing failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    main()

