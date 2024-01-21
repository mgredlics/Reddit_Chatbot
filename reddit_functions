import praw
import pandas as pd

def get_top_reddit_posts(client_id='clientID_Here',client_secret='client_secret_here',user_agent = "user_agent_here",subreddit_list='MachineLearning', limit=100, time_filter='year'):
    

    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         redirect_uri="http://localhost:8080",
                         user_agent=user_agent)
    
    posts = reddit.subreddit(subreddit_list).top(time_filter=time_filter, limit=limit)
    
    posts_df = []
    
    for post in posts:
        posts_df.append({'post_id': post.id,
                         'subreddit': post.subreddit,
                         'created_utc': post.created_utc,
                         'selftext': post.selftext,
                         'post_url': post.url,
                         'post_title': post.title,
                         'link_flair_text': post.link_flair_text,
                         'score': post.score,
                         'num_comments': post.num_comments,
                         'upvote_ratio': post.upvote_ratio
                         })
    return pd.DataFrame(posts_df)

#posts_df = get_top_reddit_posts(subreddit_list="MachineLearning+artificial", limit=100,time_filter='year')


def get_comments_for_posts_df(posts_df,client_id='client_id_here',client_secret='client_secret',user_agent = "user_agent", limit=25):
    

    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         redirect_uri="http://localhost:8080",
                         user_agent=user_agent)
    
    comments_list = []
    
    
    for post_id in posts_df['post_id']:
        # Create a submission object with the post ID
        submission = reddit.submission(post_id)
        
        submission.comments.replace_more(limit=limit)
        for comment in submission.comments.list():
            comments_list.append({'post_id': post_id, 'comment': comment.body})
            
    return pd.DataFrame(comments_list)
