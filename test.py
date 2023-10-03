import whisper
from Utils import make_Audio
from IPython.display import Audio
model = whisper.load_model("base")



def transcribe(audio):
    
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

video_name = 'sample.mp4'



make_Audio('/workspace/Structural_OCR/',video_name)
audio_dir = 'tmp_audio/'+video_name.split('.')[0]+'/' + video_name.split('.')[0]+'.wav'
result = transcribe(audio_dir)

print(result)




