import tweepy
import model_training_classification
auth = tweepy.OAuthHandler('8ENI868nIbNraM7EUqXeErSsy',
                           '2lTZWFCaDReX0hJOcLqBw4iHywKzGVCXM0kBcvhZT8GvBiO09d')
api = tweepy.API(auth)

def get_30_tweets(username): 
    
    """Takes in a Twitter username. Returns up to 100 Tweets after Twitter api is defined as 'api'."""
    
    tweets = api.user_timeline(screen_name=username, count=30, tweet_mode="extended")
    
    tweets_list = []
    
    tweets = [tweet.full_text for tweet in tweets]
    
    for tweet in tweets:
        if not tweet[:1].startswith("@") and 'https' not in tweet:
            tweets_list.append(tweet)
            
    return tweets_list

DT_Tweets = get_30_tweets('POTUS')
print(DT_Tweets)

def predict_tweets(Tweets):
    
    """Takes in a list of Tweets, cleans them, and uses the optimized_knn and optimized_rf models
       to predict the message and bias of Tweets, respectively. Returns a dataframe with the original 
       Tweets and corresponding predictions."""
    
    clean = []
    
    for tweet in Tweets:
        clean.append(clean_words(tweet))
        
    clean = [" ".join(post) for post in clean]
        
    X = vectorizer.transform(clean)

    message_preds = optimized_knn.predict(X)
    bias_preds = optimized_rf.predict(X)
    
    df = pd.DataFrame({'Tweet': Tweets, 'Partisan Bias': bias_preds, 'Message': message_preds})
    
    return df