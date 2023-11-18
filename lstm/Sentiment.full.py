import time
import requests
from datetime import datetime as dt
##2023-11-17 10:40 PM ama
# 1) TWO mode: run_once() vs. run_as_daemon()
#       ---run_once is for instant data , need to provide stock sympol 
#       ---run_as_daemon with  [predefined stock list] , for archive purpose #
#          write to csv file each day per stock
# test@test:~/testscript$ python3 ./aa/ml/Sentiment.py ftnt -v -n 3 
# ...
# Time spent on sentiment_analysis (s) 7
# Headlines processed 3, Average of Sentiment: 0.21783333333333332
# Headlines processed 3, Average of Sentiment: 0.6666666666666666
# Headlines processed 3, Average of Sentiment: 0.3333333333333333
# Headlines processed 3, Average of Sentiment: 0.3333333333333333
# 
# @todo
# 1) cmp with the stock price and plt drawing
# 2) get news from more wide , such as webpage scrawps ##
def get_headlines (company='Fortinet', lines=-1):
    # https://newsapi.org/account 349ez /Bj2008
    api_key = '2e71bb31344943829aab77e4aaf56498'
    endpoint = 'https://newsapi.org/v2/everything'

    # Search parameters
    params = {
        'apiKey': api_key,
        'q': company,
        'sortBy': 'publishedAt',
        'language': 'en',
    }   
    try:
        response = requests.get(endpoint, params=params)
        data = response.json()

        if response.status_code == 200 and data['status'] == 'ok':
            headlines = [article['title'] for article in data['articles']]
            if lines == -1:
                return headlines 
            else:
                return headlines[:lines] #           
        else:
            print(f"Error: {data['message']}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_headlines_verbose(company='Fortinet',lines=-1):
     # https://newsapi.org/account 349ez /Bj2008
    api_key = '2e71bb31344943829aab77e4aaf56498'
    endpoint = 'https://newsapi.org/v2/everything'
    """ the schema per chatgpt
            {
        "source": {
            "id": "string or null",
            "name": "string"
        },
        "author": "string or null",
        "title": "string",
        "description": "string or null",
        "url": "string",
        "urlToImage": "string or null",
        "publishedAt": "string (datetime)",
        "content": "string or null"
        } """


    # Search parameters
    params = {
        'apiKey': api_key,
        'q': company,
        'sortBy': 'publishedAt',
        'language': 'en',
    }  
    try:
        response = requests.get(endpoint, params=params)
        data = response.json()
  
        if response.status_code == 200 and data['status'] == 'ok':
            articles_info = [
                {
                    'title': article['title'],
                    'publishedAt': article['publishedAt'],
                    'content': article.get('content', '')
                } 
                for article in data.get('articles', []) 
                # data.get('articles', []): If 'articles' does not exist, it returns an empty list ([]) instead of None.\
                # You can iterate over an empty list rather than encountering a TypeError when trying to iterate over None.
            ]

            # Extract titles, publishedAt, and content simultaneously
            headlines, publishedAt, content = zip(*[
                (article['title'], article['publishedAt'], article.get('content', '')) 
                for article in articles_info
            ])
            if lines == -1:
                return headlines, publishedAt, content
            else:
                return headlines[:lines], publishedAt[:lines], content[:lines]  
        
    except Exception as e:
        print(f"Error: {e}")
import importlib
import subprocess

def install_pkg_if_not(pkg_name):
    try:
        # Try to import the package
        importlib.import_module(pkg_name)
        print(f"{pkg_name} is already installed.")
    except ImportError:
        # If the import fails, install the package using pip
        print(f"{pkg_name} is not installed. Installing...")
        subprocess.call(['pip', 'install', pkg_name])
        print(f"{pkg_name} has been installed.")

install_pkg_if_not('vaderSentiment')
install_pkg_if_not('textblob')
install_pkg_if_not('transformers')
install_pkg_if_not('flair')
# install_pkg_if_not('fasttext')  # install error, ignored
install_pkg_if_not('schedule')

def sentiment_analysis(headlines):
    """
# VADER Sentiment Analysis:
Advantages: Especially good for short texts like headlines, designed for social media sentiment analysis.
Limitations: Might not handle sarcasm or nuanced language well.
# TextBlob:
Advantages: Easy to use, suitable for simple sentiment analysis tasks.
Limitations: Might not be as accurate for complex or domain-specific language.
# Transformers (Hugging Face) - BERT Model:
Advantages: State-of-the-art models like BERT can capture intricate relationships in language, and fine-tuning allows adaptation to specific domains.
Limitations: Requires more computational resources, and fine-tuning might need a sizable labeled dataset.
# Flair:
Advantages: Provides contextual embeddings for text, can be fine-tuned for specific tasks.
Limitations: May require more effort for fine-tuning, and the default sentiment model may not be optimized for financial language.
# FastText:
Advantages: Fast and efficient, especially useful for large datasets.
Limitations: May not capture context as effectively as more advanced models like BERT.
For financial headlines, where domain-specific language and nuances are prevalent, fine-tuning a pre-trained model on financial data might lead to better results. Transformers like BERT or financial-specific models could be considered.

"""    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    from transformers import pipeline
    from flair.models import TextClassifier
    from flair.data import Sentence
    # import fasttext

    def analyze_sentiment_vader(sentence):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(sentence)
        
        # Map compound score to a numeric value
        sentiment_numeric = 1 if sentiment_scores['compound'] >= 0 else 0
        return sentiment_scores['compound'] #sentiment_numeric

    def analyze_sentiment_textblob(sentence):
        blob = TextBlob(sentence)
        
        # Map polarity to a numeric value
        sentiment_numeric = 1 if blob.sentiment.polarity >= 0 else 0
        return sentiment_numeric

    def analyze_sentiment_transformers(sentence):
        classifier = pipeline("sentiment-analysis")
        result = classifier(sentence)[0]
        
        # Map label to a numeric value
        sentiment_numeric = 1 if result['label'] == 'POSITIVE' else 0
        return sentiment_numeric

    def analyze_sentiment_flair(sentence):
        classifier = TextClassifier.load("en-sentiment")
        flair_sentence = Sentence(sentence)
        classifier.predict(flair_sentence)
        
        # Map label to a numeric value
        sentiment_label = flair_sentence.labels[0].value
        sentiment_numeric = 1 if sentiment_label == 'POSITIVE' else 0
        return sentiment_numeric

    def analyze_sentiment_fasttext(sentence, model):
        # Predict sentiment using the trained FastText model
        result = model.predict(sentence)
        
        # Map label to a numeric value
        sentiment_label = result[0][0]
        sentiment_numeric = 1 if sentiment_label == '__label__POSITIVE' else 0
        return sentiment_numeric

    # Assuming you have a trained FastText model (fasttext_model.bin)
    # fasttext_model = fasttext.load_model('fasttext_model.bin')

    # Arrays to store numeric sentiment values
    vader_sentiments = []
    textblob_sentiments = []
    transformers_sentiments = []
    flair_sentiments = []
    # fasttext_sentiments = []

    # Analyze sentiment for each headline using different tools    
    for headline in headlines:
        # VADER Sentiment Analysis
        vader_sentiments.append(analyze_sentiment_vader(headline))
        
        # TextBlob
        textblob_sentiments.append(analyze_sentiment_textblob(headline))
        
        # Transformers (Hugging Face)
        transformers_sentiments.append(analyze_sentiment_transformers(headline))
        
        # Flair
        flair_sentiments.append(analyze_sentiment_flair(headline))
        
        # FastText
        # fasttext_sentiments.append(analyze_sentiment_fasttext(headline, fasttext_model))

  
    return   vader_sentiments,    textblob_sentiments , \
            transformers_sentiments,  flair_sentiments 
       
def run(name,lines=-1, verbose=False):
    #  use it 
    
    if verbose: 
        try:
            Headlines, PublishedAt, Content = get_headlines_verbose(company=name, lines=lines)
        except TypeError as e:
            print(f"No headlines, please check company/stock name: {name}")           
            return None
        
        news_filename = 'news.'+dt.now().strftime("%Y-%m-%d")
        with open(news_filename, mode='a', newline='') as file:           
            for article_id, (headline, published_at, content) in enumerate(zip(Headlines, PublishedAt, Content), 1):
                print(f"ID: {article_id}, Headline: {headline}, Published At: \
                    {published_at}, Content: {content}",file=file)   
    else:
        Headlines = get_headlines(company=name,lines=lines)
    
    # monitor the time spent
    print('\n\n......Starting sentiment_analysis',time.ctime(time.time()))
    start_time = time.time()
    a1,a2,a3,a4 =sentiment_analysis(Headlines)
    end_time = time.time()
    if verbose:
          # Print the numeric sentiment values
        print("VADER Sentiments:", a1)
        print("TextBlob Sentiments:", a2)
        print("Transformers Sentiments:", a3)
        print("Flair Sentiments:", a4)
        # print("FastText Sentiments:", fasttext_sentiments)
    print('Time spent on sentiment_analysis (s)',int(end_time-start_time))
    
    def calculate_average(sentiments):
        total = len(sentiments)
        if total == 0:
            return 0.0  # Return 0 if the list is empty to avoid division by zero
        sum_of_sentiments = sum(sentiments)
        average = sum_of_sentiments / total
        print(f"Headlines processed {total}, Average of Sentiment: {average}")
        return average
    s = []
    for a in [a1,a2,a3,a4]:
       s.append(calculate_average(a))
    s.append(len(a1))   
    return s    
def run_as_daemon ():
    import schedule
    
    def run_sentiment_analysis():
        # Define the list of stock symbols
        stock_list = ['FTNT', 'TESLA', 'PANW', 'MICROSOFT', 'NVIDIA', 'SPCE']
        for stock in stock_list:
            s=run(name=stock,verbose=True)    
            current_date = dt.now().strftime("%Y-%m-%d")
            csv_filename = stock+'_score.csv'
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_date] + s)
    run_sentiment_analysis() # run immediately and then on schedule
    # Schedule the job to run at 3 PM each working day (Monday to Friday)
    scheduled_job=schedule.every().day.at("15:00").do(run_sentiment_analysis)
    # Display the current scheduled job
    print(f"Current Scheduled Job: {scheduled_job}")

    while True:
        # Run the scheduled jobs
        schedule.run_pending()
        # Sleep for a interval to avoid high CPU usage
        time.sleep(100)

import argparse
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment of stock')
    parser.add_argument('-s', '--server', action='store_true', help='Run as server/daemon.')
    parser.add_argument('name', type=str, nargs='?', default='', help='Specify the company name or stock symbol (optional if -s is provided).')
    parser.add_argument('-n', '--lines', type=int, default=-1, required=False, help='Specify the number of headlines. Use -1 for all headlines.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode.')

    args = parser.parse_args()

    if args.server:
        # If -s or --server option is provided, run as a server/daemon
        run_as_daemon() 
    else:
        s=run(name=args.name, lines=args.lines, verbose=args.verbose)
    

