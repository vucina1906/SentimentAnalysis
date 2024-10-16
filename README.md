This is an app that helps Booking Hotels analyze reviews and highlight causes of negative reviews so they can improve on them.

The app includes a booking review selenium scraper that I created from Strach using selenium and Chromedriver. After scraper collects all reviews and stores them in a csv file (under the name "all_reviews.csv"), the app automatically imports the csv file and analyzes it with lstm_model, which is also automatically loaded when the app is started. The model is pretrained on a dataset from the Kaggle website: https://www.kaggle.com/code/baranochnikov/sentiment-analysis-for-hotel-reviews-booking-com/input.
All analyses were done in "BookingSentimentAnalysis.ipynb," where the LSM model was created from scratch using PyTorch, and for model training I used a Google Colab GPU. After the model is loaded, it analyzes the all_reviews.csv file, and based on sentiment analysis, it will sort reviews into negative, neutral, and positive. The app will then plot those reviews but also analyze further negative reviews and plot them using WordCloud, based on word frequency in negative reviews, so later users of the app can check what was the most usual cause of negative reviews.

- For scraping, the app uses selenium. You have to download the Google Chrome browser and the accurate Chrome driver version (chromedriver.exe should be in the same folder as the app.py script). Versions must be the same. Check your Google Chrome version, and for example, if your version is 129.0.6668.101, you should download Chrome driver version 129.0.6668. Install the exact version of Selenium from the requirements.txt file. At the time you use this code, your Chrome version can be updated to a newer version from now on, so check it first
