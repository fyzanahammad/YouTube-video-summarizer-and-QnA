import streamlit as st
import os
import subprocess
from pytube import YouTube
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, BartTokenizer, BartForConditionalGeneration, BertTokenizer, BertForQuestionAnswering

# Load pre-trained models and tokenizers for summarization and QA
summarization_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
summarization_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

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
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )

    status_text.text("Transcribing audio...")
    result = pipe(wav_file_path)
    progress_bar.progress(0.6)
    return result["text"]

def generate_summary(transcription_text):
    inputs = summarization_tokenizer.encode("summarize: " + transcription_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def answer_question(transcription_text, question):
    inputs = qa_tokenizer(question, transcription_text, add_special_tokens=True, return_tensors="pt")
    answer_start_scores, answer_end_scores = qa_model(**inputs, return_dict=False)
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

def main():
    st.title("YouTube Transcription and QA App")

    youtube_url = st.text_input("Enter a YouTube video URL")

    if youtube_url:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            wav_file_path = download_audio(youtube_url, progress_bar, status_text)
            transcription_text = transcribe_audio(wav_file_path, progress_bar, status_text)
            summary = generate_summary(transcription_text)

            st.subheader("Summary")
            st.write(summary)

            st.subheader("Transcription")
            st.write(transcription_text)

            st.subheader("Ask a Question")
            question = st.text_input("Enter your question")
            if question:
                answer = answer_question(transcription_text, question)
                st.write(f"Answer: {answer}")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
