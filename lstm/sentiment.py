import time
import requests

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

def show_keyparts(company='Fortinet',lines=-1):
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
            # Modify the list comprehension to include 'description'
            articles_info = [{'title': article['title'], 'description': article['description']} for article in data['articles']]

            # Extract titles and descriptions separately if needed
            headlines = [article['title'] for article in articles_info]
            descriptions = [article['description'] for article in articles_info]

            print(headlines)
            print(descriptions)
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

    # Print the numeric sentiment values
    print("VADER Sentiments:", vader_sentiments)
    print("TextBlob Sentiments:", textblob_sentiments)
    print("Transformers Sentiments:", transformers_sentiments)
    print("Flair Sentiments:", flair_sentiments)
    # print("FastText Sentiments:", fasttext_sentiments)
    return   vader_sentiments,    textblob_sentiments , \
            transformers_sentiments,  flair_sentiments 

def main():
    #  use it 
    lines = get_headlines('FTNT', 5) # or fortinet 
    print('\n'.join(lines))

    # monitor the time spent
    print('\n\n......Starting sentiment_analysis',time.ctime(time.time()))
    start_time = time.time()
    a1,a2,a3,a4 =sentiment_analysis(lines)
    end_time = time.time()
    print('Time spent on sentiment_analysis (s)',int(end_time-start_time))
    
    def calculate_average(sentiments):
        total = len(sentiments)
        if total == 0:
            return 0.0  # Return 0 if the list is empty to avoid division by zero
        sum_of_sentiments = sum(sentiments)
        average = sum_of_sentiments / total
        print(f"Headlines processed {total}, Average of Sentiment: {average}")
        return average
    
    for a in [a1,a2,a3,a4]:
        calculate_average(a)

if __name__ == '__main__':
    main()
