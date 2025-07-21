#!/usr/bin/env python3
"""
Complete Document Summarizer AI
Reads documents (PDF, DOCX, TXT) and generates summaries using AI
"""

import os
import sys
import re
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")

# Core libraries
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import pandas as pd
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    import PyPDF2
    import docx
    import streamlit as st
    from io import BytesIO
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install transformers torch pandas nltk PyPDF2 python-docx streamlit")
    sys.exit(1)

class DocumentSummarizer:
    def __init__(self):
        """Initialize the summarizer with pre-trained models"""
        print("Loading AI models...")
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Initialize summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Initialize tokenizer for text length management
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.max_chunk_length = 1024  # BART's max input length
        
        print("AI models loaded successfully!")
    
    def read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def read_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def read_txt(self, file_path: str) -> str:
        """Read text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    def read_document(self, file_path: str) -> str:
        """Read document based on file extension"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.read_pdf(file_path)
        elif file_extension == '.docx':
            return self.read_docx(file_path)
        elif file_extension == '.txt':
            return self.read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Remove very short sentences (likely artifacts)
        sentences = sent_tokenize(text)
        filtered_sentences = [s for s in sentences if len(s.split()) > 3]
        
        return ' '.join(filtered_sentences)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks that fit model's max length"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            tokens = self.tokenizer.encode(test_chunk, add_special_tokens=True)
            
            if len(tokens) <= self.max_chunk_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Generate summary for text"""
        if not text.strip():
            return "No content to summarize."
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        if len(cleaned_text.split()) < 50:
            return "Document too short to summarize effectively."
        
        # Split into chunks if necessary
        chunks = self.chunk_text(cleaned_text)
        
        if len(chunks) == 1:
            # Single chunk - direct summarization
            summary = self.summarizer(
                chunks[0],
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return summary[0]['summary_text']
        else:
            # Multiple chunks - summarize each then combine
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Summarizing chunk {i+1}/{len(chunks)}...")
                summary = self.summarizer(
                    chunk,
                    max_length=100,
                    min_length=30,
                    do_sample=False
                )
                chunk_summaries.append(summary[0]['summary_text'])
            
            # Combine chunk summaries
            combined_summary = ' '.join(chunk_summaries)
            
            # Final summarization of combined summaries
            if len(combined_summary.split()) > 100:
                final_summary = self.summarizer(
                    combined_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return final_summary[0]['summary_text']
            else:
                return combined_summary
    
    def analyze_document(self, file_path: str) -> dict:
        """Complete document analysis"""
        try:
            # Read document
            print(f"Reading document: {file_path}")
            text = self.read_document(file_path)
            
            # Basic statistics
            word_count = len(text.split())
            sentence_count = len(sent_tokenize(text))
            
            # Generate summary
            print("Generating summary...")
            summary = self.summarize_text(text)
            
            return {
                'file_path': file_path,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'summary': summary,
                'original_text': text[:500] + "..." if len(text) > 500 else text
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'file_path': file_path
            }

def create_streamlit_app():
    """Create Streamlit web interface"""
    st.title("🤖 AI Document Summarizer")
    st.write("Upload a document (PDF, DOCX, or TXT) to get an AI-generated summary")
    
    # Initialize summarizer
    if 'summarizer' not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.summarizer = DocumentSummarizer()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=['pdf', 'docx', 'txt'],
        help="Upload PDF, DOCX, or TXT files"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # Analyze document
            with st.spinner("Analyzing document..."):
                result = st.session_state.summarizer.analyze_document(temp_path)
            
            if 'error' not in result:
                # Display results
                st.success("✅ Document analyzed successfully!")
                
                # Document info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Word Count", result['word_count'])
                with col2:
                    st.metric("Sentence Count", result['sentence_count'])
                
                # Summary
                st.subheader("📝 AI Summary")
                st.write(result['summary'])
                
                # Original text preview
                with st.expander("📄 Original Text Preview"):
                    st.text(result['original_text'])
            
            else:
                st.error(f"❌ Error: {result['error']}")
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Python script: python document_summarizer.py <document_path>")
        print("  Web interface: streamlit run document_summarizer.py")
        return
    
    file_path = sys.argv[1]
    
    # Initialize summarizer
    summarizer = DocumentSummarizer()
    
    # Analyze document
    result = summarizer.analyze_document(file_path)
    
    if 'error' not in result:
        print(f"\n📄 Document: {result['file_path']}")
        print(f"📊 Words: {result['word_count']}")
        print(f"📝 Sentences: {result['sentence_count']}")
        print(f"\n🤖 AI Summary:\n{result['summary']}")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    # Check if running with Streamlit
    if 'streamlit' in sys.modules:
        create_streamlit_app()
    else:
        main()
