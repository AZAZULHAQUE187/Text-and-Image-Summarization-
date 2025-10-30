from flask import Flask, request, render_template
from PIL import Image
import pytesseract
import re
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, BertTokenizer, BertModel
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import nltk

# Ensure necessary NLTK resources are downloaded
def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

# Call this function once at the start of your application
ensure_nltk_resources()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load models
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
gpt_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_image(image_path):
    with Image.open(image_path) as img:
        text = pytesseract.image_to_string(img)
    return text

def clean_and_split_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?]) +', text) if sentence.strip()]
    return sentences

# BERT Summarization (centrality-based)
def summarize_with_bert(sentences, top_n=5):
    def get_sentence_embedding(sentence):
        inputs = bert_tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    sentence_embeddings = torch.stack([get_sentence_embedding(sentence) for sentence in sentences])
    cosine_scores = torch.nn.functional.cosine_similarity(sentence_embeddings.unsqueeze(1), sentence_embeddings.unsqueeze(0), dim=-1)
    centrality_scores = cosine_scores.mean(dim=1).numpy()

    top_indices = np.argsort(centrality_scores)[::-1][:top_n]
    summary = "\n".join([sentences[i] for i in sorted(top_indices)])
    scores = [centrality_scores[i] for i in sorted(top_indices)]
    return summary, scores

# SBERT Summarization (centrality-based)
def summarize_with_sbert(sentences, top_n=5):
    if not sentences:
        return "No sentences found to summarize.", []
    
    sentence_embeddings = sbert_model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)
    centrality_scores = cosine_scores.mean(dim=1).cpu().numpy()

    top_indices = np.argsort(centrality_scores)[::-1][:top_n]
    summary = "\n".join([sentences[i] for i in sorted(top_indices)])
    scores = [centrality_scores[i] for i in sorted(top_indices)]
    return summary, scores

# GPT-based summarization
def summarize_with_gpt(text, max_length=150, min_length=40):
    summary_chunks = gpt_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    summary = summary_chunks[0]['summary_text']
    
    # Split the summary into sentences while keeping it clean
    sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?]) +', summary) if sentence.strip()]
    
    # Ensure scores are calculated correctly based on number of sentences
    scores = [1 / (i + 1) for i in range(len(sentences))] if sentences else []
    
    # Join sentences with newline characters for better formatting
    formatted_summary = "\n".join(sentences)
    
    return formatted_summary, scores

# TextRank Summarization
def summarize_with_text_rank(text, top_n=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count=top_n)
    
    summary = "\n".join(str(sentence) for sentence in summary_sentences)
    scores = [1 / (i + 1) for i in range(len(summary_sentences))]
    return summary, scores

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No image part'
        
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'
        
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            
            # Extract and clean text from image
            text = extract_text_from_image(image_path)
            sentences = clean_and_split_text(text)
            full_text = " ".join(sentences)

            # Generate summaries with all models
            bert_summary, bert_scores = summarize_with_bert(sentences)
            sbert_summary, sbert_scores = summarize_with_sbert(sentences)
            gpt_summary, gpt_scores = summarize_with_gpt(full_text)
            text_rank_summary, text_rank_scores = summarize_with_text_rank(full_text)

            # Calculate word counts
            original_word_count = len(full_text.split())
            bert_word_count = len(bert_summary.split())
            sbert_word_count = len(sbert_summary.split())
            gpt_word_count = len(gpt_summary.split())
            text_rank_word_count = len(text_rank_summary.split())

            return render_template('upload.html', text=text, 
                                   original_word_count=original_word_count,
                                   bert_summary=bert_summary, bert_scores=bert_scores,
                                   bert_word_count=bert_word_count,
                                   sbert_summary=sbert_summary, sbert_scores=sbert_scores,
                                   sbert_word_count=sbert_word_count,
                                   gpt_summary=gpt_summary, gpt_scores=gpt_scores,
                                   gpt_word_count=gpt_word_count,
                                   text_rank_summary=text_rank_summary, text_rank_scores=text_rank_scores,
                                   text_rank_word_count=text_rank_word_count)

    return render_template('upload.html')

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    model = request.form.get('model')
    rating = request.form.get(f'{model}_rating')

    # Here you can save the feedback to a database or a file
    with open('feedback.txt', 'a') as f:
        f.write(f'Model: {model}, Rating: {rating}\n')

    # Optionally, redirect back to the main page or show a success message
    return 'Thank you for your feedback!'

if __name__ == '__main__':
    app.run(debug=True)
