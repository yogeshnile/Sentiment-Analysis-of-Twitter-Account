# %%
import pandas as pd
import numpy as np
import tweepy 
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import re 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# %%
"""
### Read Twitter API
"""

# %%
log = pd.read_csv("api keys.csv")

# %%
consumerKey = log["key"][0]
consumerSecret = log["key"][1]
accessToken = log["key"][2]
accessTokenSecret = log["key"][3]

# %%
"""
### Authenticate API keys

"""

# %%
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
authenticate.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(authenticate, wait_on_rate_limit = True) # api object

# %%
"""
### Get a Tweet from Twitter
"""

# %%
post = api.user_timeline(screen_name="elonmusk", count = 100, lang ="en", tweet_mode="extended")

# %%
"""
####  Print the last 5 tweets
"""

# %%
i=1
for tweet in post[:5]:
    print(str(i) +') '+ tweet.full_text + '\n')
    i= i+1

# %%
"""
### Save Tweets in DataFrame
"""

# %%
twitter = pd.DataFrame([tweet.full_text for tweet in post], columns=['Tweets'])

# %%
"""
### Clean a Tweet
"""

# %%
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    return text

# %%
twitter['Tweets'] = twitter['Tweets'].apply(cleanTxt)

# %%
twitter.head()

# %%
"""
## Apply a Tweet for get subjectivity and polarity
"""

# %%
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# %%
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# %%
twitter['Subjectivity'] = twitter['Tweets'].apply(getSubjectivity)
twitter['Polarity'] = twitter['Tweets'].apply(getPolarity)

# %%
twitter

# %%
"""
### find a common word of tweets and analysis using ploting
"""

# %%
allWords = ' '.join([twts for twts in twitter['Tweets']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# %%
"""
### Apply a Sentiment in word and save in datafame
"""

# %%
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# %%
twitter['Analysis'] = twitter['Polarity'].apply(getAnalysis)

# %%
twitter

# %%
twitter.info()

# %%
"""
### Get positive tweets
"""

# %%
positive = twitter.loc[twitter['Analysis'].str.contains('Positive')]
positive.drop(['Subjectivity','Polarity'], axis=1, inplace=True)

# %%
positive.head()

# %%
positive.shape

# %%
"""
### Get negative Tweets
"""

# %%
negative = twitter.loc[twitter['Analysis'].str.contains('Negative')]
negative.drop(['Subjectivity','Polarity'], axis=1, inplace=True)

# %%
negative.head()

# %%
negative.shape

# %%
"""
### Get Neutral Tweets
"""

# %%
neutral = twitter.loc[twitter['Analysis'].str.contains('Neutral')]
neutral.drop(['Subjectivity','Polarity'], axis=1, inplace=True)

# %%
neutral.head()

# %%
neutral.shape

# %%
plt.figure(figsize=(8,6))
for i in range(0, twitter.shape[0]):
    plt.scatter(twitter["Polarity"][i], twitter["Subjectivity"][i], color='Blue')
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.savefig('tweets analysis.png', dir=300)
plt.show()

# %%
"""
### Persentage of positive, ngative and neutral tweets
"""

# %%
print(str(round((positive.shape[0]/twitter.shape[0])*100, 1))+' %')

# %%
print(str(round((negative.shape[0]/twitter.shape[0])*100, 1))+' %')

# %%
twitter['Analysis'].value_counts()

# %%
"""
### Plotting and visualizing tweets
"""

# %%
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
twitter['Analysis'].value_counts().plot(kind = 'bar')
plt.savefig('Sentiment Analysis.jpeg', dir=300)
plt.show()


# %%
