from flask import Flask, render_template, request
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time
import torch
import torch.nn as nn
import pickle
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

app = Flask(__name__)

# Define the LSTM model architecture
class SentimentLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(hidden[-1])  # Use last hidden state for prediction
        return output

# Load the LSTM model and vocab globally
vocab = None
model = None
tokenizer = None

def load_model_and_vocab():
    global vocab, model, tokenizer
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 64
    output_dim = 1

    model = SentimentLSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('lstm_model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode

    tokenizer = get_tokenizer("basic_english")

# Function to scrape reviews from all pages
def scrape_all_pages_reviews(url):
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(5)

    reviews = []
    page = 1

    while True:
        print(f"Scraping page {page}...")
        review_blocks = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="review"]')

        if not review_blocks:
            print(f"No reviews found on page {page}.")
            break

        print(f"Found {len(review_blocks)} reviews on page {page}.")

        for review_block in review_blocks:
            try:
                title = review_block.find_element(By.CSS_SELECTOR, 'h3[data-testid="review-title"]').text
            except:
                title = "No title"

            try:
                translation_button = review_block.find_element(By.XPATH, '//span[text()="Show translation"]/..')
                translation_button.click()
                time.sleep(1)  # Wait for the translation to load
                text = review_block.find_element(By.CSS_SELECTOR, 'div[data-testid="review-positive-text"]').text
                print("Translated review found!")
            except:
                try:
                    text = review_block.find_element(By.CSS_SELECTOR, 'div[data-testid="review-positive-text"]').text
                except:
                    text = "No review text"

            reviews.append({
                'review_title': title,
                'review_text': text
            })

        try:
            next_button = driver.find_element(By.XPATH, '//button[@aria-label="Next page"]')
            is_disabled = next_button.get_attribute("disabled")
            print(f"Next button disabled: {is_disabled}")
            if is_disabled is not None:
                print("Reached the last page. Stopping.")
                break
            next_button.click()
            page += 1
            time.sleep(5)
        except Exception as e:
            print(f"Error finding next page button or clicking it: {e}")
            break

    driver.quit()
    df_reviews = pd.DataFrame(reviews)
    df_reviews.to_csv('all_reviews.csv', index=False, encoding='utf-8-sig')
    print("Scraping complete! Reviews saved to 'all_reviews.csv'.")
    return df_reviews

# Function to predict sentiment for each review
def predict_sentiment(review_title, review_text):
    review = review_title + "." + review_text
    tokenized_review = [vocab[token] for token in tokenizer(review) if token in vocab]
    review_tensor = torch.tensor(tokenized_review, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        predicted_rating = model(review_tensor).item()
    if predicted_rating <= 3:
        return "negative"
    elif 4 <= predicted_rating <= 6:
        return "neutral"
    else:
        return "positive"

# Function to plot sentiment distribution
def plot_sentiment_distribution(df_reviews):
    sentiment_counts = df_reviews['predicted_sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'yellow', 'red'])
    plt.title('Sentiment Distribution of Reviews')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('static/sentiment_distribution.png')
    plt.close()

# Function to highlight common words in negative reviews
def highlight_negative_words(df_reviews):
    common_words_set = set([
        "the", "is", "and", "a", "to", "in", "of", "for", "on", "it",
        "this", "that", "are", "with", "as", "at", "was", "be", "by",
        "an", "but", "or", "from", "not", "no", "yes", "very", "nice",
        "clean", "location", "review", "text"
    ])

    negative_reviews = df_reviews[df_reviews['predicted_sentiment'] == 'negative']
    all_negative_words = ' '.join(negative_reviews['review_text'].tolist()).split()

    filtered_negative_words = [word for word in all_negative_words if word.lower() not in common_words_set]
    common_negative_words = Counter(filtered_negative_words).most_common(10)
    return common_negative_words

def create_wordcloud(df_reviews):
    negative_reviews = df_reviews[df_reviews['predicted_sentiment'] == 'negative']
    all_negative_words = ' '.join(negative_reviews['review_text'].tolist())
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_negative_words)
    wordcloud.to_file('static/negative_wordcloud.png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        url = request.form['review_url']
        df_reviews = scrape_all_pages_reviews(url)

        sentiments = []
        for index, row in df_reviews.iterrows():
            sentiment = predict_sentiment(row['review_title'], row['review_text'])
            sentiments.append(sentiment)
        df_reviews['predicted_sentiment'] = sentiments
        df_reviews.to_csv('all_reviews_with_sentiment.csv', index=False, encoding='utf-8-sig')

        plot_sentiment_distribution(df_reviews)
        create_wordcloud(df_reviews)  # Generate WordCloud image

        return render_template('results.html', 
                               sentiment_image='sentiment_distribution.png', 
                               wordcloud_image='negative_wordcloud.png')


if __name__ == '__main__':
    load_model_and_vocab()  # Load model and vocabulary at startup
    app.run(debug=True)
