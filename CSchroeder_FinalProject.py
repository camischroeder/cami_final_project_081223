import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from collections import Counter

posts = pd.read_csv("/Users/camilaschroeder/Desktop/DATA/FinalProject/CSchroeder_FinalProject/raw_database/the-antiwork-subreddit-dataset-posts.csv")

col_drop2 = ["subreddit.id", "subreddit.name", "subreddit.nsfw", "type", "domain"]
posts.drop(columns=col_drop2, inplace = True)

names_col = {"id": "post_id", "score": "votes", "created_utc": "date", "selftext" : "text"}
posts.rename(columns= names_col, inplace= True)

comments = pd.read_csv("/Users/camilaschroeder/Desktop/DATA/FinalProject/CSchroeder_FinalProject/raw_database/the-antiwork-subreddit-dataset-comments.csv")

comments = comments.dropna()

new_names = {"id":"user_id", "subreddit.name":"subreddit", "created_utc":"date", "body":"comments", "score":"votes"}
comments.rename(columns = new_names, inplace = True)

col_drop = ["type", "subreddit", "subreddit.id", "subreddit.nsfw"]
comments.drop(columns = col_drop, inplace = True)

comments["post_id"] = comments["permalink"].str[43:49]

average_sentiment = comments.groupby("post_id")["sentiment"].mean().reset_index()
posts = pd.merge(posts, average_sentiment, on="post_id", suffixes=('', '_avg'))

comments["engagement"] = comments["votes"].abs()
posts["engagement"] = posts["votes"].abs()
comm_engagement_by_post = comments.groupby("post_id")["engagement"].sum().reset_index()
posts = pd.merge(posts, comm_engagement_by_post, on="post_id", how="left")
posts["engagement"] = posts["engagement_x"] + posts["engagement_y"]
posts.drop(columns = ["engagement_y", "engagement_x"], inplace=True)

col1 = ["post_id", "title", "text", "url", "votes", "sentiment", "engagement", "date", "permalink"]
posts = posts[col1]

posts["date"] = pd.to_datetime(posts["date"], unit = "s")

col2 = ["user_id", "post_id", "comments", "votes", "sentiment", "engagement", "date", "permalink"]
comments = comments[col2]

comments["date"] = pd.to_datetime(comments["date"], unit = "s")

posts["title"] = posts["title"].str.lower()
posts["text"] = posts["text"].str.lower()
comments["comments"] = comments["comments"].str.lower()

def filter_by_year(df, start_year, end_year=None):
    if end_year is None:
        end_year = start_year
    filtered_content = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
    return filtered_content

def filter_by_month_year(df, start_month_year, end_month_year=None):
    if end_month_year is None:
        end_month_year = start_month_year
    start_year, start_month = map(int, start_month_year.split('-'))
    end_year, end_month = map(int, end_month_year.split('-'))
    filtered_content = df[
        (df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year) &
        (df['date'].dt.month >= start_month) & (df['date'].dt.month <= end_month)
    ]
    return filtered_content

def generate_wordcloud(df, key_list):
    df["combined_text"] = df["text"] + " " + df["title"]
    combined_text = " ".join(df["combined_text"].dropna())
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=set(STOPWORDS.union(key_list))).generate(combined_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def generate_wordcloud2(df, key_list):
    combined_text = " ".join(df["comments"].dropna())

    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=set(STOPWORDS.union(key_list))).generate(combined_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def generate_combined_wordcloud(df1, df2, key_list):
    df1["combined_text"] = df1["text"] + " " + df1["title"]
    combined_text = " ".join(df1["combined_text"].dropna()) + " ".join(df2["comments"].dropna())

    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=set(STOPWORDS.union(key_list))).generate(combined_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def count_mentions(df1, df2, key_list, subject):
    df1["combined_text"] = df1["text"] + " " + df1["title"]
    masks1 = [df1['combined_text'].str.contains(word, case=False, na=False) for word in key_list]
    combined_mask1 = pd.concat(masks1, axis=1).any(axis=1)
    count_mentions1 = combined_mask1.sum()

    masks2 = [df2['comments'].str.contains(word, case=False, na=False) for word in key_list]
    combined_mask2 = pd.concat(masks2, axis=1).any(axis=1)
    count_mentions2 = combined_mask2.sum()

    print(f"{subject} mentioned {count_mentions1 + count_mentions2} times")

def count_mentions2(row, keyword_list):
    return sum(keyword in str(row).lower() for keyword in keyword_list)

stopwords2 = ["work", "job", "jobs", "people", "working", "delete", "deleted", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", 
              "day", "night", "thing", "https", "etc", "s", "do", "don't", "even"]

stopwords1 = set(STOPWORDS.union(stopwords2))

covid_words = ["covid", "pandemic", "corona", "virus"]

quiet_quit = ["quiet quitting", "disengagement", "passive resistance", "burnout", "withdraw", "apathy", "laying flat"]

anarchism_list = ["anarchism", "anarchist", "anarchy", "anarco", "stateless", "libertarian", "communalism", "anti-capitalism"]

capitalism_list = ["capital", "capitalism", "capitalist", "free market", "private property", "consumerism", "invisible hand"]

comm_socialist_list = ["communism", "socialism", "soviet", "marxist", "leninist", "proletariat", "labor", "labour", "revolution", "means of production", "marx",
                       "engels", "stalin", "lenin", "trotski"]

fascism_list = ["fascism", "fascist", "nazism", "nazi", "hitler", "mussolini", "trump", "bolsonaro", "orban", "le pen", "antifa", "authoritarianism", "totalitarianism", 
                "nationalism", "ultra-nationalism", "populism", "dictatorship", "autocracy", "anti-democratic", "xenophobia", "white supremacy", "authoritatian populism"]

neg_list = ["sad", "unhappy", "sorrow", "grief", "angry", "anger", "furious", "irritated", "outraged", "enraged", "fear", "scared", "frightened", "anxious", "terrified",
            "disgust", "repulsed", "loathe", "nauseated", "revulsion", "anxiety", "anxious", "nervous", "worried", "stress", "aprehens", ]

republican = ["trump", "mcconnell", "republican", "gop", "boebert", "marjorie taylor greene", "ted cruz", "kevin mccarthy"]

posts_13_19 = filter_by_year(posts, 2013, 2019)
comments_13_19 = filter_by_year(comments, 2013, 2019)
posts_20_22 = filter_by_month_year(posts, "2020-01", "2022-02")
comments_20_22 = filter_by_month_year(comments, "2020-01", "2022-02")

generate_combined_wordcloud(posts_13_19, comments_13_19, stopwords1)
generate_combined_wordcloud(posts_20_22, comments_20_22, stopwords1)
generate_combined_wordcloud(posts, comments, stopwords1)

count_mentions(posts, comments, covid_words, "Covid")
count_mentions(posts, comments, anarchism_list, "Anarchism")
count_mentions(posts, comments, capitalism_list, "Capitalism")
count_mentions(posts, comments, comm_socialist_list, "Communism/Socialism")
count_mentions(posts, comments, fascism_list, "Fascism")

comments.fillna(" ", inplace=True)
posts.fillna(" ", inplace=True)

posts["combined_text"] = posts["title"] + " " + posts["text"]

posts["covid"] = posts["combined_text"].apply(lambda row: count_mentions2(row, covid_words))
comments["covid"] = comments["comments"].apply(lambda row: count_mentions2(row, covid_words))

posts["fascism"] = posts["combined_text"].apply(lambda row: count_mentions2(row, fascism_list))
comments["fascism"] = comments["comments"].apply(lambda row: count_mentions2(row, fascism_list))

posts["anarchism"] = posts["combined_text"].apply(lambda row: count_mentions2(row, anarchism_list))
comments["anarchism"] = comments["comments"].apply(lambda row: count_mentions2(row, anarchism_list))

posts["capitalism"] = posts["combined_text"].apply(lambda row: count_mentions2(row, capitalism_list))
comments["capitalism"] = comments["comments"].apply(lambda row: count_mentions2(row, capitalism_list))

posts["comm_soc"] = posts["combined_text"].apply(lambda row: count_mentions2(row, comm_socialist_list))
comments["comm_soc"] = comments["comments"].apply(lambda row: count_mentions2(row, comm_socialist_list))

posts["neg"] = posts["combined_text"].apply(lambda row: count_mentions2(row, neg_list))
comments["neg"] = comments["comments"].apply(lambda row: count_mentions2(row, neg_list))

posts["republican"] = posts["combined_text"].apply(lambda row: count_mentions2(row, republican))
comments["republican"] = comments["comments"].apply(lambda row: count_mentions2(row, republican))

posts["quiet"] = posts["combined_text"].apply(lambda row: count_mentions2(row, quiet_quit))
comments["quiet"] = comments["comments"].apply(lambda row: count_mentions2(row, quiet_quit))

posts['date'] = posts['date'].astype(str)
posts = posts[posts["date"] != " "]
posts["date"] = pd.to_datetime(posts["date"], errors='coerce')

comments['date'] = comments['date'].astype(str)
comments = comments[comments["date"] != " "]
comments["date"] = pd.to_datetime(comments["date"], errors='coerce')

posts_top10 = posts.sort_values(by="votes", ascending=False)
posts_top10 = posts_top10.head(10)

comments_top100 = comments.sort_values(by="votes", ascending=False)
comments_top100 = comments_top100.head(100)

with pd.ExcelWriter("top_quali.xlsx") as writer:
    posts_top10.to_excel(writer, sheet_name="posts_top10", index=False)
    comments_top100.to_excel(writer, sheet_name="comments_top100", index=False)

comments2 = comments.groupby("post_id")[["votes", "sentiment", "engagement", "covid", "fascism", "anarchism", "capitalism", "comm_soc", "neg",
                                         "republican", "quiet"]].sum()
comments2['active_users'] = comments.groupby('post_id')['user_id'].nunique()

posts2 = pd.merge(posts, comments2, on="post_id", how="outer", suffixes=("_posts", "_comments"))
posts2.dropna(subset=["votes_posts"], inplace=True)

posts2["votes"] = posts2["votes_comments"] + posts2["votes_posts"]
posts2["engagement"] = posts2["engagement_comments"] + posts2["engagement_posts"]
posts2["sentiment"] = posts2["sentiment_comments"] + posts2["sentiment_posts"]
posts2["covid"] = posts2["covid_comments"] + posts2["covid_posts"]
posts2["fascism"] = posts2["fascism_comments"] + posts2["fascism_posts"]
posts2["anarchism"] = posts2["anarchism_comments"] + posts2["anarchism_posts"]
posts2["capitalism"] = posts2["capitalism_comments"] + posts2["capitalism_posts"]
posts2["comm_soc"] = posts2["comm_soc_comments"] + posts2["comm_soc_posts"]
posts2["neg"] = posts2["neg_posts"] + posts2["neg_comments"]
posts2["republican"] = posts2["republican_posts"] + posts2["republican_comments"]
posts2["quiet"] = posts2["quiet_comments"] + posts2["quiet_posts"]

dropcol = ["votes_comments", "votes_posts", "engagement_comments", "engagement_posts", "sentiment_comments", "sentiment_posts",
           "covid_comments", "covid_posts", "fascism_comments", "fascism_posts", "anarchism_comments",
           "anarchism_posts", "capitalism_comments", "capitalism_posts", "comm_soc_posts", "comm_soc_comments", "neg_posts",
           "neg_comments", "republican_posts", "republican_comments", "quiet_posts", "quiet_comments"]
posts2.drop(columns=dropcol, inplace=True)

posts2.to_csv("postsfin.csv", index=False)
comments.to_csv("commentsfin.csv", index=False)
