import pandas as pd
from transformers import pipeline, BertTokenizer
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import threading
from tqdm import tqdm
import torch

# Инициализация переменной для DataFrame
df = None

def load_csv_file():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Success", "File loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

def split_text_into_chunks(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

device = 0 if torch.cuda.is_available() else -1
nlp = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", device=device)
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def get_answer_from_text(text, question):
    chunks = split_text_into_chunks(text)
    answers_with_confidence = []

    for chunk in tqdm(chunks, desc="Processing chunks", leave=False):
        response = nlp({'context': chunk, 'question': question})
        confidence = response.get('score', 0)
        answers_with_confidence.append((response['answer'], confidence))

    return max(answers_with_confidence, key=lambda x: x[1])

def process_question(question):
    if df is None:
        messagebox.showerror("Error", "Please load a CSV file first.")
        return

    answers_for_all_texts = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):
        context = row['text']
        best_answer = get_answer_from_text(context, question)
        answers_for_all_texts.append({
            "text_index": index,
            "best_answer": best_answer[0],
            "confidence": best_answer[1]
        })

    top_10_answers = sorted(answers_for_all_texts, key=lambda x: x['confidence'], reverse=True)[:10]
    
    for answer_info in top_10_answers:
        answers_text.insert(tk.END, f"Text Index: {answer_info['text_index']}, Best Answer: {answer_info['best_answer']}, Confidence: {answer_info['confidence']:.4f}\n")

def show_text_index():
    try:
        text_index = int(text_index_entry.get())
        if df is not None and 0 <= text_index < len(df):
            messagebox.showinfo("Text Index", f"Text Index: {text_index}\nText: {df.loc[text_index, 'text']}")
        else:
            messagebox.showerror("Error", "Invalid text index.")
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid integer for the text index.")

def on_submit():
    question = question_entry.get()
    if question:
        answers_text.delete(1.0, tk.END)
        threading.Thread(target=process_question, args=(question,)).start()

# Создание GUI
root = tk.Tk()
root.title("Question Answering with BERT")

load_button = tk.Button(root, text="Load CSV File", command=load_csv_file)
load_button.pack()

question_label = tk.Label(root, text="Enter your question:")
question_label.pack()

question_entry = tk.Entry(root, width=50)
question_entry.pack()

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack()

answers_label = tk.Label(root, text="Top 10 Best Answers:")
answers_label.pack()

answers_text = scrolledtext.ScrolledText(root, width=80, height=20)
answers_text.pack()

text_index_label = tk.Label(root, text="Enter text index:")
text_index_label.pack()

text_index_entry = tk.Entry(root, width=10)
text_index_entry.pack()

show_text_index_button = tk.Button(root, text="Show Text", command=show_text_index)
show_text_index_button.pack()

root.mainloop()
