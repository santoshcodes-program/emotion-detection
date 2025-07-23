
🧠 Emotion Detection using CNN + GloVe

This project predicts emotions like **joy**, **sadness**, **anger**, and more from short text inputs using Natural Language Processing (NLP) and a deep learning **CNN model with GloVe embeddings**.

It uses the `dair-ai/emotion` dataset and includes complete preprocessing, model training, evaluation, and a **Gradio-based user interface** for interactive testing.

🎯 Goal: Achieve 92%+ validation accuracy — and we did!



 😃 Emotion Classes

The model classifies input text into one of the following six emotions:

```
['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
```

---

## 🗂 Project Structure

```
📁 emotion-detection-project/
├── train.ipynb            # Main training notebook
├── best_model.pt          # Trained PyTorch CNN model
├── glove.6B.100d.txt      # Pre-trained GloVe embeddings
├── predict.py             # Inference script
├── vocab.pkl              # Saved vocabulary/index mapping
├── README.md              # Project documentation
├── demo.png               # Screenshot of UI (optional)
```

---

⚙ How It Works
 
🔄 Preprocessing

* Lowercased all text
* Removed special characters
* Tokenized sentences
* Converted words to indices using vocab of top 10,000 words
* Applied padding/truncation to fixed length (60 tokens)

📐 Embeddings

* Used **GloVe 100d pre-trained word vectors**
* Constructed embedding matrix and fed into model

🧠 Model Architecture — TextCNN

* Embedding layer (GloVe 100d)
* 3 Conv1D layers with filter sizes \[3, 4, 5]
* MaxPooling over time
* Dropout for regularization
* Fully Connected Softmax output layer

🔍 Inference Flow

1. Input text is cleaned and tokenized
2. Words are mapped to indices, padded
3. Input is passed to the trained CNN model
4. Predicted emotion label is returned

---

🛠 Setup Instructions

Clone the repo

```bash
git clone https://github.com/your-username/emotion-detection-project.git
cd emotion-detection-project
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run the App

```bash
python ui_gradio.py
```

📊 Dataset Used

* **Dataset:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
* **Samples:** \~20,000
* **Classes:** 6 emotion categories

---

👨‍💻 Author

Krishna Sai Santhosh and Jayanthi Venupusa
GitHub: [santosh & jayanthi](https://github.com/santoshcodes-program)
Project guided by: Personal learning + DAIR AI inspiration

