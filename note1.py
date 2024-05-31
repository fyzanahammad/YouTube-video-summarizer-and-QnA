

from youtube_transcript_api import YouTubeTranscriptApi

def get_video_id_from_url(url):
    # This function extracts the video ID from the YouTube URL
    if "youtube.com/watch?v=" in url:
        return url.split("watch?v=")[1]
    else:
        raise ValueError("Invalid YouTube URL")

def get_transcript(youtube_url):
    try:
        # Extract the video ID from the URL
        video_id = get_video_id_from_url(youtube_url)

        # Fetch the available transcripts
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Print each entry of the transcript
        for entry in transcript:
            print(f"Start: {entry['start']} - Duration: {entry['duration']}s")
            print(entry['text'] + '\n')

    except Exception as e:
        print(f"An error occurred: {e}")

youtube_url = "https://www.youtube.com/watch?v=ysLiABvVos8"
get_transcript(youtube_url)



import os
import subprocess
from pytube import YouTube

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

# Example usage:
youtube_url = "https://www.youtube.com/watch?v=ysLiABvVos8"
wav_file_path = download_audio(youtube_url)


import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

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
print(result["chunks"])

data = result["chunks"]

for entry in data:
    start_time, end_time = entry['timestamp']
    text = entry['text']
    print(f"Timestamp: {start_time:.2f} - {end_time:.2f}")
    print(f"Text: {text}")

result = pipe(wav_file_path, return_timestamps=False)
print(result)

print(result['text'])




