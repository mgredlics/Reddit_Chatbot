"""
Created on Sun Jan 21 14:37:19 2024

@author: mgredlics

Idea from Thu Vu data analytics youtube
"""

import sys
sys.path.append("/home/mgredlics/Python/Reddit_Chatbot_Final/Functions")
import pandas as pd
import numpy as np
import datetime as dt
from reddit_functions import get_top_reddit_posts, get_comments_for_posts_df

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline

# Helps to bypass the token limit
from llama_index import SimpleDirectoryReader,LLMPredictor,PromptHelper,VectorStoreIndex,Document,load_index_from_storage, StorageContext
from langchain_community.chat_models import ChatOpenAI
import os

openAIkey_yahoo = 'openAIkey_here'

posts_df = get_top_reddit_posts(subreddit_list='Denver', limit=100, time_filter='week')
posts_df.to_csv('/home/mgredlics/Python/Reddit_Chatbot_Final/ML_posts.csv', header=True, index=False)

# Get comments on all posts
comments_df = get_comments_for_posts_df(posts_df,limit=25)
comments_df.to_csv('/home/mgredlics/Python/Reddit_Chatbot_Final/ML_comments.csv', header=True, index=False)

## Read previous files

posts_df = pd.read_csv('/home/mgredlics/Python/Reddit_Chatbot_Final/ML_posts.csv')
comments_df = pd.read_csv('/home/mgredlics/Python/Reddit_Chatbot_Final/ML_comments.csv')

# Convert created date to normal datetime
posts_df['created_date'] = posts_df['created_utc'].apply(lambda x: dt.datetime.fromtimestamp(x))
posts_df['created_year'] = posts_df['created_date'].dt.year
posts_df['created_month'] = posts_df['created_date'].dt.month

# Now merge comments file with posts file
comments_posts_df = posts_df.merge(comments_df,on='post_id', how='left')

# Remove rows with missing comments
comments_posts_df = comments_posts_df[-comments_posts_df['comment'].isnull()]

# Generate Wordcloud on post title

post_title_text = ' '.join([title for title in posts_df['post_title'].str.lower()])

word_cloud = WordCloud(collocation_threshold=2, width=1000,height=500,
                       background_color='white').generate(post_title_text)

plt.figure(figsize=(10,5))
plt.imshow(word_cloud)
plt.axis("off")


# Generate Wordcloud on post comments

post_title_text = ' '.join([title for title in comments_posts_df['comment'].str.lower()])

word_cloud = WordCloud(collocation_threshold=2, width=1000,height=500,
                       background_color='white').generate(post_title_text)

plt.figure(figsize=(10,5))
plt.imshow(word_cloud)
plt.axis("off")


# Let's try sentiment analysis - github


sentiment_classifier = pipeline(model = "finiteautomata/bertweet-base-sentiment-analysis")

sentiment_classifier("She is beautiful")

def get_sentiment(text):
    try:
        sentiment = sentiment_classifier(text)[0]['label']
    except:
        sentiment = 'Not classified'
    
    return sentiment


## Try to get emotion
emotion_classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion',return_all_scores=True)

def get_emotion(text):
    # Get emotion prediction scores
    pred_scores = emotion_classifier(text)
    
    # Get emotion with highest prediction score
    emotion = max(pred_scores[0], key=lambda x: x['score'])['label']
    
    return emotion

def get_emotion_confidence(text):
    # Get emotion prediction scores
    pred_scores = emotion_classifier(text)
    
    # Get emotion with highest prediction score
    confidence = max(pred_scores[0], key=lambda x: x['score'])['score']
    
    return confidence
    
# Filter out comments with a specific phrase
comments_posts_df_sub = comments_posts_df[comments_posts_df['comment'].str.contains('homeless')]

# Run through sentiment
comments_posts_df_sub['sentiment'] = comments_posts_df_sub['comment'].astype(str).apply(lambda x: get_sentiment(x))


comments_posts_df_sub['trunc_comment'] = comments_posts_df_sub['comment'].astype(str).str[:510]
comments_posts_df_sub['emotion'] = comments_posts_df_sub['trunc_comment'].astype(str).apply(lambda x: get_emotion(x))
comments_posts_df_sub['confidence'] = comments_posts_df_sub['trunc_comment'].astype(str).apply(lambda x: get_emotion_confidence(x))


plt.figure(figsize = (20, 30))
plt.subplot(5, 2, 1)
plt.gca().set_title('Sentiment for the Topic')
sns.countplot(x = 'sentiment', palette = 'Set2', data = comments_posts_df_sub)
plt.subplot(5, 2, 2)
plt.gca().set_title('Emotion for the Topic')
sns.countplot(x = 'emotion', palette = 'Set2', data = comments_posts_df_sub)
plt.subplot(5, 2, 3)
plt.gca().set_title('Emotion Confidence for the Topic')
sns.histplot(x = 'confidence', kde = False, data = comments_posts_df_sub)


# Combine all posts and comments so comments are together. One row per post and save to a text file
comments_posts_df_tmp = comments_posts_df[['post_title','selftext','comment']].astype(str)
agg_comments = comments_posts_df_tmp.groupby(['post_title','selftext'])['comment'].apply('. '.join).reset_index()

agg_comments['combined_text'] = agg_comments.astype(str).agg('. '.join,axis=1)
all_text = ' '.join(agg_comments['combined_text'])

# Save file to txt file
f = open("/home/mgredlics/Python/Reddit_Chatbot_Final/textdata/all_text_reddit.txt","w")
f.write(all_text)
f.close


directory_path = "/home/mgredlics/Python/Reddit_Chatbot_Final/textdata/"
# Build Index
def construct_index(directory_path):
    # set maximum input size
    context_window = 3900
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 0.1
    # set chunk size limit
    chunk_size_limit = 600
    
    # define LLM (ChatGPT gpt-3.5-turbo)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    prompt_helper = PromptHelper(context_window = context_window, num_output=num_outputs, chunk_overlap_ratio=max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = VectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, PromptHelper=prompt_helper)
    
    #index = VectorStoreIndex(documents, llm_predictor=llm_predictor,PromptHelper=prompt_helper)


    index.storage_context.persist("/home/mgredlics/Python/Reddit_Chatbot_Final/index.json")
    
    return index
    





# Build Chatbot
def ask_me_anything(question):
    # Rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="/home/mgredlics/Python/Reddit_Chatbot_Final/index.json")
    # Load index from the storage context
    new_index = load_index_from_storage(storage_context)
    
    #index = VectorStoreIndex.from_documents('/home/mgredlics/Python/Reddit_Chatbot/index')
    #index = VectorStoreIndex.load_from_disk('/home/mgredlics/Python/Reddit_Chatbot/index.json')
    
    new_query_engine = new_index.as_query_engine()
    response = new_query_engine.query(question)
    print(response)





###### Run Chatbot

os.environ["OPENAI_API_KEY"] = openAIkey_yahoo

construct_index("/home/mgredlics/Python/Reddit_Chatbot_Final/textdata/")
ask_me_anything("What should I do in Denver this weekend")
