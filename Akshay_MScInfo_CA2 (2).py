#installed below packages in anaconda
#conda install -c conda-forge tweepy
#conda install -c conda-forge textblob
#conda install -c anaconda scikit-learn

#Most commom error occuers in working code while using tweepy are 
#1-TweepError: Twitter error response: status code = 401
#It means that your credentials (in this case, your consumer/access tokens) are invalid. Try re-creating the credentials
#2-RateLimitError: [{'message': 'Rate limit exceeded', 'code': 88}]
#When you reached the API streaming limit ,so this error will resolve after an hour 


import tweepy as tw#for accessing twitter api
import pandas as pd #for data manipulation
import matplotlib.pyplot as plt#for data visualization
import seaborn as sn#for data visualization based on matplotlin
import re#for regular expressiong and using search function which returns the object of scanned text
from textblob import TextBlob#for processsing textual data
from sklearn.feature_extraction.text import TfidfVectorizer #machine learning library for text classification and vectorization


# Authenticate to Twitter
auth = tw.OAuthHandler("7REkVKufOJ3H7xKwHK9I0uI9r","GlHAVnjXYJyOFAvv2RMiDpe4g1zfmOybGIpRKwvHkqeKfub3ZK")
auth.set_access_token("357911706-smoA7FPvYi1fWTj37BWIs0bCHfXjucwTqtdkJT5z", "2frLqr69t6Lp3cmRwoYgz4QLXMeqjmldPL6q70OFincly")
#when we reach the Twitter Streaming API limit. It takes about one hour to extract tweets again.so it will not fail so use wait on rate limit
api = tw.API(auth,wait_on_rate_limit=True)#Create API object 
# confirm account
print(">>>>>>Feature No-1<<<<<")
print("--------------------------")
print("Confirming Account")

print("--------------------------")
print ("API NAME IS: ", api.me().name)
print(" ")

#Timeline Tweets
print(" ")
print("--------------------------")
print(">>>>>>Feature No-2<<<<<")
print("--------------------------")
print("My 5 Timeline Tweets")
print(" ")
Timeline_tweets= api.home_timeline(count=5)
for tweet in Timeline_tweets:
    print (tweet.text)

 #for tweet in public_tweets:
print(" ")
print("--------------------------")
print(">>>>>>Feature No-3<<<<<")
print("--------------------------")
print("Followers count of 5 Tweets on Timeline")
Timeline_tweets= api.home_timeline(count=5)
for tweet in Timeline_tweets:
    print('The followers Count is-> ' + str(tweet.user.followers_count))

#Find the count of followers of specific user
print(" ")
print("--------------------------")
print(">>>>>>Feature No-4<<<<<")
print("--------------------------")
print("Screen name and Followers count of Specific User:elonmusk")
user = api.get_user('elonmusk')
print('Screen Name-->' + user.screen_name)
print("Count-->" + str(user.followers_count))


print(" ")
print("--------------------------")
print(">>>>>>Feature No-5<<<<<")
print("--------------------------")
print("Source and Source url of tweets")
print(" ")

public_tweets= api.home_timeline();
for tweet in public_tweets:
   print ('Source-->'+ tweet.source)
   print ('Source URL-->'+ tweet.source_url)


print(" ")
print("--------------------------")
print(">>>>>>Feature No-6<<<<<")
print("--------------------------")
print("Different Keyword Analysis using search method")
print(" ")
#for tweet in public_tweets:
date_since = "2020-05-05"# specific date tweets 
tweets = tw.Cursor(api.search,
              q="Covid 19",
              lang="en", since=date_since).items(5)

print("Analyzed keyword Covid 19 in Tweets--")
print(" ")
i=1
#for loop for printing tweets with index number using str()
for tweet in tweets:
    print(str(i) + ')'+ tweet.text + '\n')
    i = i + 1

# Iterate and print tweets
#Search with "Ireland" Cand Collect tweets
tweets = tw.Cursor(api.search,
              q="Ireland",
              lang="en",since=date_since).items(5)
print("Analyzed keyword 'Ireland'in Tweets--")
print(" ")
i=1
for tweet in tweets:
    print(str(i) + ')'+ tweet.text + '\n')
    i = i + 1

#Search with "Data Mining" Cand Collect tweets
tweets = tw.Cursor(api.search,
              q="Data Mining",
              lang="en",since=date_since).items(5)
print("Analyzed keyword 'Data Mining'in Tweets--")
print(" ")
i=1
for tweet in tweets:
    print(str(i) + ')'+ tweet.text + '\n')
    i = i + 1
 
#Search with "Holiday" Cand Collect tweets
tweets = tw.Cursor(api.search,
              q="Holiday",
              lang="en",since=date_since).items(5)
print("Analyzed keyword 'Holiday'in Tweets--")
print(" ")
i=1
for tweet in tweets:
    print(str(i) + ')'+ tweet.text + '\n')
    i = i + 1
    

print(" ")
print("--------------------------")
print(">>>>>>Feature No-7<<<<<")
print("--------------------------")
print("Crreating Data Frame for search keyword Covid 19")
print(" ")
print("Analyzed keyword Ireland--")

tweets = tw.Cursor(api.search,
              q="Covid 19",
              lang="en").items(5)

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
print(data)


#tweet, screena name, user location,followers count ,source ,source url and tweet creation details functions alltogether
print(" ")
print("--------------------------")
print("--------------------------")
print("Different tweepy function for search keyword Covid 19 Tweet")
print(" ")
tweets = tw.Cursor(api.search,
              q="Covid 19",
              lang="en").items(10)

data_req = [[tweet.text,tweet.user.screen_name, tweet.user.location,
             tweet.user.screen_name, tweet.user.followers_count,tweet.user.created_at] for tweet in tweets]

#set dataframe size because of multiple rows and columns
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)

tweet_text = pd.DataFrame(data=data_req, 
                    columns=['Tweet','User', "Location", 'Screen Name', 'No of Followers','Tweeted On'])
print(tweet_text)

#
print(" ")
print("--------------------------")
print("--------------------------")
print("Different tweepy function for search keyword Ireland Tweet")
print(" ")
tweets = tw.Cursor(api.search,
              q="Covid 19",
              lang="en").items(10)

data_req = [[tweet.text,tweet.user.screen_name, tweet.user.location,
             tweet.user.screen_name, tweet.user.followers_count,tweet.user.created_at] for tweet in tweets]

#set dataframe size because of multiple rows and columns
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)

tweet_text = pd.DataFrame(data=data_req, 
                    columns=['Tweet','User', "Location", 'Screen Name', 'No of Followers','Tweeted On'])
print(tweet_text)

print(" ")
print("--------------------------")
print("--------------------------")
print("Different tweepy function for search keyword Data Mining Tweet")
print(" ")
tweets = tw.Cursor(api.search,
              q="Covid 19",
              lang="en").items(10)

data_req = [[tweet.text,tweet.user.screen_name, tweet.user.location,
             tweet.user.screen_name, tweet.user.followers_count,tweet.user.created_at] for tweet in tweets]

#set dataframe size because of multiple rows and columns
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)

tweet_text = pd.DataFrame(data=data_req, 
                    columns=['Tweet','User', "Location", 'Screen Name', 'No of Followers','Tweeted On'])
print(tweet_text)

print(" ")
print("--------------------------")
print("--------------------------")
print("Different tweepy function for search keyword Holiday Tweet")
print(" ")
tweets = tw.Cursor(api.search,
              q="Covid 19",
              lang="en").items(10)

data_req = [[tweet.text,tweet.user.screen_name, tweet.user.location,
             tweet.user.screen_name, tweet.user.followers_count,tweet.user.created_at] for tweet in tweets]

#set dataframe size because of multiple rows and columns
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)

tweet_text = pd.DataFrame(data=data_req, 
                    columns=['Tweet','User', "Location", 'Screen Name', 'No of Followers','Tweeted On'])
print(tweet_text)

#Sentiments Analysis 
#Clean up 100 recent tweets. For this analysis, you only need to remove URLs from the tweets.
print(" ")
print("--------------------------")
print(">>>>>>Feature No-8<<<<<")
print(" ")
print("Analyze Sentiment in Tweets for Keyword 'Covid 19'" )
print(" ")
print("---Positve if value >0, Negative if value<0, Neutral if value=0---" )

#remove url from the tweets
#Replace URLs found in a text string with nothing (i.e. it will remove the URL from the string).

def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
#The same txt string with url's removed

tweets = tw.Cursor(api.search,
              q="Covid 19",
              lang="en").items(100)

tweets_no_urls = [TextBlob(remove_url(tweet.text)) for tweet in tweets]

tweets_no_urls[:5]

# Create list of polarity valuesx and tweet text
C_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in tweets_no_urls]

# Create dataframe containing polarity values and tweet text
C_sent_df = pd.DataFrame(C_values, columns=["polarity", "tweet"])
C_sent_df = C_sent_df[C_sent_df.polarity != 0]

C_sent_df.head()

#For TextBlog, if the polarity is >0, it is positive, <0 -is negative and ==0 is neutral.
fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
C_sent_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
        ax=ax, color="purple")

plt.title("Sentiments from Tweets on the 'Covid 19'")
plt.show()

print(" ")
print("--------------------------")
print(" ")
print("Analyze Sentiment in Tweets for Keyword 'Ireland'" )
print(" ")
print("---Positve if value >0, Negative if value<0, Neutral if value=0---" )


def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

tweets = tw.Cursor(api.search,
              q="Ireland",
              lang="en").items(100)

all_tweets_no_urls = [TextBlob(remove_url(tweet.text)) for tweet in tweets]

all_tweets_no_urls[:5]

I_sent_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in all_tweets_no_urls]

# Create dataframe containing polarity values and tweet text
I_sent_df = pd.DataFrame(I_sent_values, columns=["polarity", "tweet"])
I_sent_df = I_sent_df[I_sent_df.polarity != 0]

I_sent_df.head()

#For TextBlog, if the polarity is >0, it is positive, <0 -is negative and ==0 is neutral.
fig, ax = plt.subplots(figsize=(8, 6))

I_sent_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
        ax=ax, color="purple")

plt.title("Sentiments from Tweets on the 'Ireland'")
plt.show()

print(" ")
print("--------------------------")
print(" ")
print("Analyze Sentiment in Tweets for Keyword 'Data Mining'" )
print(" ")
print("---Positve if value >0, Negative if value<0, Neutral if value=0---" )

def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

tweets = tw.Cursor(api.search,
              q="Data Mining",
              lang="en").items(100)

all_tweets_no_urls = [TextBlob(remove_url(tweet.text)) for tweet in tweets]

all_tweets_no_urls[:5]

DM_sent_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in all_tweets_no_urls]

# Create dataframe containing polarity values and tweet text
DM_sent_df = pd.DataFrame(DM_sent_values, columns=["polarity", "tweet"])
DM_sent_df = DM_sent_df[DM_sent_df.polarity != 0]

DM_sent_df.head()

#For TextBlog, if the polarity is >0, it is positive, <0 -is negative and ==0 is neutral.
fig, ax = plt.subplots(figsize=(8, 6))

DM_sent_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
        ax=ax, color="purple")

plt.title("Sentiments from Tweets on the 'Ireland'")
plt.show()

print(" ")
print("--------------------------")
print(" ")
print("Analyze Sentiment in Tweets for Keyword 'Holiday'" )
print(" ")
print("---Positve if value >0, Negative if value<0, Neutral if value=0---" )

def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

tweets = tw.Cursor(api.search,
              q="Holiday",
              lang="en").items(100)

all_tweets_no_urls = [TextBlob(remove_url(tweet.text)) for tweet in tweets]

all_tweets_no_urls[:5]

H_sent_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in all_tweets_no_urls]

# Create dataframe containing polarity values and tweet text
H_sent_df = pd.DataFrame(H_sent_values, columns=["polarity", "tweet"])
H_sent_df = H_sent_df[H_sent_df.polarity != 0]

H_sent_df.head()

#For TextBlog, if the polarity is >0, it is positive, <0 -is negative and ==0 is neutral.
fig, ax = plt.subplots(figsize=(8, 6))

H_sent_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
        ax=ax, color="purple")

plt.title("Sentiments from Tweets on the 'Ireland'")
plt.show()
#plotting the username frequency of tweets from home timeline   

print(" ")
print("--------------------------")
print(">>>>>>Feature No-9<<<<<")
print("--------------------------")
print("Bargraph of Most tweets from the different users on timeline")
print(" ")
 
home_tweet = tw.Cursor(api.home_timeline)
tweets = [i.author.screen_name for i in home_tweet.items(200)]
fig, ax = plt.subplots(figsize=(8, 6))
unq = set(tweets)
freq = {uname: tweets.count(uname) for uname in unq}
ax.set_xlabel('Tweeted Users')
ax.set_ylabel('No of Tweets')
plt.bar(range(len(unq)), freq.values())
plt.show()


print(" ")
print("--------------------------")
print(">>>>>>Feature No-10<<<<<")
print("--------------------------")
print("Confusion Matrix or Statsitical Analysis with Data visualization(Heatmap)")
print("--------------------------")
print("Confusion Matrix for keyword= Covid 19")
print(" ")

api = tw.API(auth)
tweets = tw.Cursor(api.search,
              q="Covid 19",
              lang="en").items(10)

data = [[tweet.user.screen_name, tweet.user.followers_count] for tweet in tweets]


df = pd.DataFrame(data, columns=['Actual','Predicted'])

#compute a simple cross-tabulation of two (or more) factors
confusion_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'])
#print (confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
plt.show()

print(" ")
print("--------------------------")
print('Confusion Matrix for keyword= Ireland')
print(" ")
api = tw.API(auth)
tweets = tw.Cursor(api.search,
              q="Ireland",
              lang="en").items(10)

data = [[tweet.user.screen_name, tweet.user.followers_count] for tweet in tweets]


df = pd.DataFrame(data, columns=['Actual','Predicted'])

#compute a simple cross-tabulation of two (or more) factors
confusion_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'])
#print (confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
plt.show()

print(" ")
print("--------------------------")
print('Confusion Matrix for keyword= Data Mining')
print(" ")
api = tw.API(auth)
tweets = tw.Cursor(api.search,
              q="Data Mining",
              lang="en").items(10)

data = [[tweet.user.screen_name, tweet.user.followers_count] for tweet in tweets]


df = pd.DataFrame(data, columns=['Actual','Predicted'])

#compute a simple cross-tabulation of two (or more) factors
confusion_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'])
#print (confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
plt.show()


print(" ")
print("--------------------------")
print('Confusion Matrix for keyword= Holiday')
print(" ")
api = tw.API(auth)
tweets = tw.Cursor(api.search,
              q="Holiday",
              lang="en").items(10)

data = [[tweet.user.screen_name, tweet.user.followers_count] for tweet in tweets]


df = pd.DataFrame(data, columns=['Actual','Predicted'])

#compute a simple cross-tabulation of two (or more) factors
confusion_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'])
#print (confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
plt.show

#basic Text feature extraction
#Convert text to a set of representative numerical value
print(" ")
print("------Text Feature--------")
print(">>>>>>Feature No-11<<<<<")
print("--------------------------")
print("Convert text to a set of representative numerical value")
print(" ")
sample_tweets =['BREAKING: 48 patients recover from COVID-19 in Lagos',
          'You will account for COVID-19 donations',' Lagos Assembly tells Sanwo-Olu']
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)

vec = TfidfVectorizer()
X = vec.fit_transform(sample_tweets)
d = pd.DataFrame(X.toarray(), columns=vec.get_feature_names()) 
print(d)

#most automatic mining of social media data relies on some form of encoding the text as numbers
#Encoding data is by word counts: you take each snippet of text, count the occurrences of each word within it, and put the results in a table
print(" ")
print("------Text Feature--------")
print(">>>>>>Feature No-12<<<<<")
print("--------------------------")
print("Convert text to a set of representative numerical value")
print(" ")
print("Text feature for Keyword Covid 19")
public_tweets= api.home_timeline();
tweets = tw.Cursor(api.search,
              q="Covid 19",
              lang="en").items(4)

sample_tweets = [tweet.text for tweet in tweets]

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)

#vectorization
vec = TfidfVectorizer()
X = vec.fit_transform(sample_tweets)
d = pd.DataFrame(X.toarray(), columns=vec.get_feature_names()) 
print(d)

print(" ")
print("--------------------------")
print("Convert text to a set of representative numerical value")
print(" ")
print("Text feature for Keyword Ireland")

public_tweets= api.home_timeline();
tweets = tw.Cursor(api.search,
              q="Ireland",
              lang="en").items(4)

sample_tweets = [tweet.text for tweet in tweets]

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)

#vectorization
vec = TfidfVectorizer()
X = vec.fit_transform(sample_tweets)
d = pd.DataFrame(X.toarray(), columns=vec.get_feature_names()) 
print(d)

print(" ")
print("--------------------------")
print("Convert text to a set of representative numerical value")
print(" ")
print("Text feature for Keyword Data Mining")
public_tweets= api.home_timeline();
tweets = tw.Cursor(api.search,
              q="Data Mining",
              lang="en").items(4)

sample_tweets = [tweet.text for tweet in tweets]

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
#vectorization

vec = TfidfVectorizer()
X = vec.fit_transform(sample_tweets)
d = pd.DataFrame(X.toarray(), columns=vec.get_feature_names()) 
print(d)

print(" ")
print("--------------------------")
print("Convert text to a set of representative numerical value")
print(" ")
print("Text feature for Keyword Holiday") 
public_tweets= api.home_timeline();
tweets = tw.Cursor(api.search,
              q="Holiday",
              lang="en").items(4)

sample_tweets = [tweet.text for tweet in tweets]

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
#vectorization

vec = TfidfVectorizer()
X = vec.fit_transform(sample_tweets)
d = pd.DataFrame(X.toarray(), columns=vec.get_feature_names()) 
print(d)