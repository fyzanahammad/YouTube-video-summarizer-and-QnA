import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import os
import subprocess
from pytube import YouTube
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def get_video_id_from_url(url):
    if "youtube.com/watch?v=" in url:
        return url.split("watch?v=")[1]
    else:
        raise ValueError("Invalid YouTube URL")

def rename_audio(file_path):
    # Define the new file name
    new_file_name = "audio_file"

    # Get the directory of the file
    file_directory = os.path.dirname(file_path)

    # Get the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Create a unique new file name
    count = 1
    while os.path.exists(os.path.join(file_directory, f"{new_file_name}_{count}{file_extension}")):
        count += 1

    new_file_name = f"{new_file_name}_{count}{file_extension}"
    # Get the new file path
    new_file_path = os.path.join(file_directory, new_file_name)

    # Rename the file
    os.rename(file_path, new_file_path)

    print(f"Renamed '{file_path}' to '{new_file_path}'")

    return new_file_path

def convert_to_wav(file_path):
    # Define output directory
    output_directory = os.path.dirname(file_path)

    # Define the new file name with .wav extension
    wav_file_name = os.path.splitext(os.path.basename(file_path))[0] + ".wav"

    # Define the output path for the converted .wav file
    wav_file_path = os.path.join(output_directory, wav_file_name)

    # Execute FFmpeg command to convert to .wav format
    subprocess.run(["ffmpeg", "-i", file_path, "-acodec", "pcm_s16le", "-ar", "44100", wav_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return wav_file_path

def download_audio(youtube_url):
    # Create a YouTube object with the URL
    yt = YouTube(youtube_url)

    # Define output directory
    output_path = "downloaded_audio"

    # Get the best audio stream
    audio_stream = yt.streams.get_audio_only()

    # Download the audio stream
    downloaded_file_path = audio_stream.download(output_path)

    # Rename the downloaded audio file
    renamed_file_path = rename_audio(downloaded_file_path)

    print(f"Downloaded '{audio_stream.title}' to '{renamed_file_path}' as '{audio_stream.mime_type}'.")

    # Convert to .wav format
    wav_file_path = convert_to_wav(renamed_file_path)

    print(f"Converted '{renamed_file_path}' to '{wav_file_path}'.")

    return wav_file_path

def transcribe_audio(wav_file_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-tiny"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(wav_file_path, return_timestamps=True)
    return result["chunks"]

def main():
    st.title("YouTube Transcription App")

    youtube_url = st.text_input("Enter a YouTube video URL")

    if youtube_url:
        try:
            video_id = get_video_id_from_url(youtube_url)
            wav_file_path = download_audio(youtube_url)
            transcription_data = transcribe_audio(wav_file_path)

            st.subheader("Transcription")
            for entry in transcription_data:
                start_time, end_time = entry['timestamp']
                text = entry['text']
                st.write(f"Timestamp: {start_time:.2f} - {end_time:.2f}")
                st.write(f"Text: {text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()