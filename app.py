import os
from dotenv import load_dotenv
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Get Groq API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput input, .stTextArea textarea {
        border-radius: 20px;
        padding: 10px 15px;
    }
    .stButton button {
        border-radius: 20px;
        background: linear-gradient(145deg, #6c5ce7, #a363d9);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(108,92,231,0.4);
    }
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .response-area {
        background: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin-top: 15px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

class VideoProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )

    def get_video_id(self, url):
        if "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        return url.split("v=")[1].split("&")[0]

    def get_available_languages(self, video_id):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            languages = []
            
            # Get manual transcripts
            for transcript in transcript_list._manually_created_transcripts.values():
                languages.append({
                    'code': transcript.language_code,
                    'name': transcript.language,
                    'type': 'Manual'
                })
            
            # Get auto-generated transcripts
            for transcript in transcript_list._generated_transcripts.values():
                languages.append({
                    'code': transcript.language_code,
                    'name': transcript.language,
                    'type': 'Auto-generated'
                })
            
            return languages
        except Exception as e:
            return []

    def get_transcript(self, url, target_language='en'):
        try:
            video_id = self.get_video_id(url)
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            try:
                # First try to get transcript in target language
                transcript = transcript_list.find_transcript([target_language])
            except:
                # If target language not available, try auto-generated transcript
                try:
                    transcript = transcript_list.find_generated_transcript([target_language])
                except:
                    # If no auto-generated transcript in target language, try manual transcript and translate
                    try:
                        transcript = transcript_list.find_manually_created_transcript()
                    except:
                        # Get any available transcript and translate
                        available_transcripts = (
                            list(transcript_list._manually_created_transcripts.values()) +
                            list(transcript_list._generated_transcripts.values())
                        )
                        if not available_transcripts:
                            raise Exception("No transcripts available")
                        transcript = available_transcripts[0]
                
                if target_language != transcript.language_code:
                    transcript = transcript.translate(target_language)
            
            # Get transcript text
            transcript_text = " ".join([t['text'] for t in transcript.fetch()])
            
            # Try to get video title, but don't fail if we can't
            try:
                yt = YouTube(url)
                title = yt.title
            except:
                title = f"Video ({video_id})"
            
            return transcript_text, title
            
        except Exception as e:
            available_langs = self.get_available_languages(video_id)
            if available_langs:
                lang_info = "\nAvailable languages:\n" + "\n".join([
                    f"- {lang['name']} ({lang['code']}) [{lang['type']}]"
                    for lang in available_langs
                ])
            else:
                lang_info = "\nNo transcripts available for this video."
                
            st.error(f"Error fetching transcript: {str(e)}{lang_info}")
            return None, None

class ContentAnalyzer:
    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            self.embeddings = None
        self.vector_store = None
        self.combined_text = ""
        self.video_titles = []

    def process_videos(self, urls, language='en'):
        if not self.embeddings:
            st.error("Embeddings not initialized. Cannot process videos.")
            return None

        processor = VideoProcessor()
        all_text = []
        self.video_titles = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, url in enumerate(urls):
            status_text.markdown(f"📹 Processing video {i+1}/{len(urls)}...")
            text, title = processor.get_transcript(url, target_language=language)
            if text:
                all_text.append(text)
                self.video_titles.append(title or f"Video {i+1}")
            progress_bar.progress((i+1)/len(urls))
        
        if not all_text:
            st.error("No valid transcripts found from the provided URLs.")
            return None

        self.combined_text = "\n\n".join(all_text)
        chunks = processor.text_splitter.split_text(self.combined_text)
        
        if not chunks:
            st.error("No text chunks generated from the transcripts.")
            return None

        try:
            metadatas = []
            for i, _ in enumerate(chunks):
                video_index = i % len(self.video_titles)
                metadatas.append({"source": self.video_titles[video_index]})
            
            self.vector_store = FAISS.from_texts(chunks, self.embeddings, metadatas=metadatas)
            return self.vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None

    def query_content(self, question, model_name="mixtral-8x7b-32768"):
        if not self.vector_store:
            return "Please process videos first!"
        
        retriever = self.vector_store.as_retriever(search_kwargs={'k': 4})
        llm = ChatGroq(temperature=0.5, model_name=model_name, api_key=GROQ_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": question})
        return result['result'], result['source_documents']

    def generate_summary(self, model_name="mixtral-8x7b-32768"):
        llm = ChatGroq(temperature=0.2, model_name=model_name, api_key=GROQ_API_KEY)
        prompt = f"""
        Generate a comprehensive summary of the following content from multiple YouTube videos.
        Use markdown formatting with sections and bullet points.
        Content: {self.combined_text[:10000]}
        """
        return llm.invoke(prompt).content

    def generate_timeline(self, model_name="mixtral-8x7b-32768"):
        llm = ChatGroq(temperature=0.3, model_name=model_name, api_key=GROQ_API_KEY)
        prompt = f"""
        Create a chronological timeline of key events from this content:
        {self.combined_text[:10000]}
        Format as markdown with timestamps and brief descriptions.
        """
        return llm.invoke(prompt).content

# Streamlit App Layout
def main():
    st.title("🎬 VideoMind Analyzer")
    st.markdown("### AI-Powered YouTube Video Analysis Suite")

    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ContentAnalyzer()

    # Sidebar for Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        model_name = st.selectbox(
            "Choose AI Model",
            ["mixtral-8x7b-32768", "llama2-70b-4096"],
            help="Select Groq model for processing"
        )
        
        language = st.selectbox(
            "Transcript Language",
            ["en", "te", "hi", "ta", "ml", "kn", "es", "fr", "de", "ja", "ko", "zh-Hans"],
            format_func=lambda x: {
                'en': 'English',
                'te': 'Telugu',
                'hi': 'Hindi',
                'ta': 'Tamil',
                'ml': 'Malayalam',
                'kn': 'Kannada',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'ja': 'Japanese',
                'ko': 'Korean',
                'zh-Hans': 'Chinese (Simplified)'
            }[x],
            help="Select the language for video transcripts. If not available, will be translated to this language."
        )
        
        st.markdown("---")
        st.markdown("**How to use:**\n1. Paste YouTube URLs\n2. Process videos\n3. Ask questions or generate insights")
        st.markdown("---")
        st.markdown("Made with ❤️ by Fyzan Ahammad")

    # Main Content Area
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            urls = st.text_area(
                "📥 Enter YouTube URLs (one per line)",
                height=100,
                placeholder="Paste YouTube links here..."
            )
        with col2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("🚀 Process Videos", use_container_width=True):
                if urls.strip():
                    url_list = [url.strip() for url in urls.split('\n') if url.strip()]
                    st.session_state.analyzer.process_videos(url_list, language)
                    st.success("✅ Videos processed successfully!")
                else:
                    st.error("Please enter at least one YouTube URL")


    # Features Container
    if st.session_state.analyzer.combined_text:
        with st.expander("💡 Analysis Features", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 Generate Summary", use_container_width=True):
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.analyzer.generate_summary(model_name)
            
            with col2:
                if st.button("⏳ Create Timeline", use_container_width=True):
                    with st.spinner("Creating timeline..."):
                        timeline = st.session_state.analyzer.generate_timeline(model_name)
            
            with col3:
                if st.button("🔑 Extract Key Terms", use_container_width=True):
                    with st.spinner("Identifying key terms..."):
                        llm = ChatGroq(temperature=0.3, model_name=model_name, api_key=GROQ_API_KEY)
                        terms = llm.invoke(f"Extract 15-20 key terms from this content: {st.session_state.analyzer.combined_text[:10000]}").content
            
            if 'summary' in locals():
                st.markdown(summary)
            if 'timeline' in locals():
                st.markdown(timeline)
            if 'terms' in locals():
                st.markdown(terms)

        # Question Answering Section
        st.markdown("---")
        question = st.text_input(
            "💬 Ask anything about the video content:",
            placeholder="Type your question here..."
        )
        
        if question:
            with st.spinner("🔍 Analyzing content..."):
                answer, sources = st.session_state.analyzer.query_content(question, model_name)
                
                # Display Answer
                with st.container(border=True):
                    st.markdown(f"**Answer:**\n{answer}")
                    
                    # Display Sources
                    with st.expander("📚 View Source Context"):
                        for doc in sources:
                            st.markdown(f"```\n{doc.page_content}\n```")
                            if 'source' in doc.metadata:
                                st.caption(f"Source: {doc.metadata['source']}")
                            else:
                                st.caption("Source: Unknown")

if __name__ == "__main__":
    main()