import streamlit as st
import os
import subprocess
from pytube import YouTube
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, BertModel, BertTokenizer
import re

# Load pre-trained model and tokenizer for semantic similarity
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def rename_audio(file_path):
    new_file_name = "audio_file"
    file_directory = os.path.dirname(file_path)
    file_extension = os.path.splitext(file_path)[1]
    count = 1
    while os.path.exists(os.path.join(file_directory, f"{new_file_name}_{count}{file_extension}")):
        count += 1
    new_file_name = f"{new_file_name}_{count}{file_extension}"
    new_file_path = os.path.join(file_directory, new_file_name)
    os.rename(file_path, new_file_path)
    return new_file_path

def convert_to_wav(file_path, status_text):
    status_text.text("Converting audio to WAV format...")
    output_directory = os.path.dirname(file_path)
    wav_file_name = os.path.splitext(os.path.basename(file_path))[0] + ".wav"
    wav_file_path = os.path.join(output_directory, wav_file_name)
    subprocess.run(["ffmpeg", "-i", file_path, "-acodec", "pcm_s16le", "-ar", "44100", wav_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_file_path

def download_audio(youtube_url, progress_bar, status_text):
    status_text.text("Downloading audio from YouTube...")
    yt = YouTube(youtube_url)
    output_path = "downloaded_audio"
    audio_stream = yt.streams.get_audio_only()
    downloaded_file_path = audio_stream.download(output_path)
    renamed_file_path = rename_audio(downloaded_file_path)
    progress_bar.progress(0.2)
    return convert_to_wav(renamed_file_path, status_text)

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def calculate_similarity(sentence1, sentence2):
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return cosine_similarity.item()

def reassemble_sentences(transcription_data):
    new_transcription_data = []
    buffer_sentence = ""
    buffer_start_time = None
    buffer_end_time = None

    for entry in transcription_data:
        start_time, end_time = entry['timestamp']
        text = entry['text']
        
        # If the current sentence buffer is empty, initialize it
        if not buffer_sentence:
            buffer_sentence = text
            buffer_start_time = start_time
            buffer_end_time = end_time
        else:
            # If the current buffer sentence does not end with a full stop, append the current text
            if not buffer_sentence.endswith('.'):
                buffer_sentence += " " + text
                buffer_end_time = end_time
            else:
                new_transcription_data.append({
                    'timestamp': (buffer_start_time, buffer_end_time),
                    'text': buffer_sentence.strip()
                })
                buffer_sentence = text
                buffer_start_time = start_time
                buffer_end_time = end_time

    # Add the last buffered sentence
    if buffer_sentence:
        new_transcription_data.append({
            'timestamp': (buffer_start_time, buffer_end_time),
            'text': buffer_sentence.strip()
        })

    return new_transcription_data

def adjust_timestamps(transcription_data, similarity_threshold=0.85, progress_bar=None, status_text=None):
    new_chunks = []
    buffer_sentence = ""
    buffer_start_time = None
    buffer_end_time = None

    total_chunks = len(transcription_data)
    for idx, entry in enumerate(transcription_data):
        start_time, end_time = entry['timestamp']
        text = entry['text']
        
        # Ensure text is split only at full stops
        sentences = re.split(r'(?<=\.)\s+', text)

        for sentence in sentences:
            if not buffer_sentence:
                buffer_sentence = sentence
                buffer_start_time = start_time
                buffer_end_time = end_time
            else:
                similarity = calculate_similarity(buffer_sentence, sentence)
                if similarity >= similarity_threshold:
                    buffer_sentence += " " + sentence
                    buffer_end_time = end_time
                else:
                    new_chunks.append({
                        'timestamp': (buffer_start_time, buffer_end_time),
                        'text': buffer_sentence.strip()
                    })
                    buffer_sentence = sentence
                    buffer_start_time = start_time
                    buffer_end_time = end_time

        if progress_bar:
            progress_bar.progress(0.6 + 0.4 * (idx + 1) / total_chunks)
        if status_text:
            status_text.text(f"Processing transcription data... ({idx + 1}/{total_chunks})")

    if buffer_sentence:
        new_chunks.append({
            'timestamp': (buffer_start_time, buffer_end_time),
            'text': buffer_sentence.strip()
        })

    # Ensure timestamps are continuous and non-overlapping
    for i in range(1, len(new_chunks)):
        new_chunks[i]['timestamp'] = (new_chunks[i - 1]['timestamp'][1], new_chunks[i]['timestamp'][1])


    return new_chunks

def transcribe_audio(wav_file_path, progress_bar, status_text):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-tiny"

    status_text.text("Loading transcription model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
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

    status_text.text("Transcribing audio...")
    result = pipe(wav_file_path, return_timestamps=True, generate_kwargs={"language": "english"})
    progress_bar.progress(0.6)
    return result["chunks"]

def main():
    st.title("YouTube Transcription App")

    youtube_url = st.text_input("Enter a YouTube video URL")

    if youtube_url:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            wav_file_path = download_audio(youtube_url, progress_bar, status_text)
            transcription_data = transcribe_audio(wav_file_path, progress_bar, status_text)
            reassembled_data = reassemble_sentences(transcription_data)
            adjusted_transcription_data = adjust_timestamps(reassembled_data, progress_bar=progress_bar, status_text=status_text)
            progress_bar.progress(1.0)
            status_text.text("Transcription and processing complete.")

            st.subheader("Transcription")
            for entry in adjusted_transcription_data:
                start_time, end_time = entry['timestamp']
                text = entry['text']
                st.write(f"Timestamp: {start_time:.2f} - {end_time:.2f}")
                st.write(f"Text: {text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
