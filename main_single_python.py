# %%
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import numpy as np
import ast
from collections import Counter, defaultdict
%matplotlib inline

# %% [markdown]
# Read-in datasets containing userId's, ratings, and Overviews

# %%
with open("./ratings_small.csv", encoding='utf-8') as f:
    ratings_df = pd.read_csv(f).astype({"movieId": int})

with open("./movies_metadata.csv", encoding='utf-8') as f:
    movie_details_df = pd.read_csv(f)

# %% [markdown]
# Clean "id" column by removing nulls and converting column to an integer

# %%
# Remove rows that are not numeric values (there were a few dates in the dataset)
movie_details_df['id'] = pd.to_numeric(movie_details_df['id'], errors='coerce')
movie_details_df = movie_details_df.dropna(subset=['id']).astype({"id": int})
print(movie_details_df.columns)

# %% [markdown]
# Merge the two dataframes to get Users, Ratings, and Movie Overviews in the same Dataframe. We also do some quick exploratory analysis to understand the dataset better

# %%
# Merge the two dataframes on the movieId column
df_movies_with_ratings = pd.merge(ratings_df, movie_details_df, left_on="movieId", right_on="id", how="inner")
display(df_movies_with_ratings.describe()) # Get a High-level look at the datset

df_movie_filter = df_movies_with_ratings[['userId','movieId','rating','genres','overview','title']].fillna('')
display(df_movie_filter.head(100)) # Show a couple examples of the data

# Count the number of distinct users and movies
num_unique_titles = df_movie_filter['title'].nunique()
num_unique_users = df_movie_filter['userId'].nunique()
print(f"Number of distinct titles: {num_unique_titles}")
print(f"Number of distinct users: {num_unique_users}")

# Let's get a sense of volume for the most "active" users
user_title_counts = df_movie_filter.groupby('userId')['title'].nunique()
filtered_counts = user_title_counts[user_title_counts > 200] # Chose 200 because that looked to be where a manageable cutoff was for visualizing

# Plot the number of movies viewed for the most active users
plt.figure(figsize=(12, 6))
plt.bar(filtered_counts.index.astype(str), filtered_counts.values)
plt.xlabel('userId')
plt.ylabel('Number of Different Movies Viewed')
plt.title('Users Who Have Watched More Than 200 Different Movies')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

display(df_movie_filter[df_movie_filter['userId'] == 564]['title']) # Example

# %% [markdown]
# Given we are trying to use LDA as a form of "genre", we want to understand how many and what kind of genres are in the dataset.

# %%
# "Genres" are held in dictionaries within a column of the DataFrame, so we need to go one level further to work with those values
genre_ids = df_movies_with_ratings['genres'].dropna().apply(
    lambda x: [genre['id'] for genre in eval(x)] if isinstance(x, str) else []
) # Get Genre Id's

genre_names = df_movies_with_ratings['genres'].dropna().apply(
    lambda x: [genre['name'] for genre in eval(x)] if isinstance(x, str) else []
) # Get Genre names

# Convert into a single list of id's and names
all_genre_ids = [genre_id for sublist in genre_ids for genre_id in sublist]
all_genre_names = [genre_name for sublist in genre_names for genre_name in sublist]

# Count the number of unique id's
unique_genre_count = len(set(all_genre_ids))

print(f"Number of unique genre IDs: {unique_genre_count}")
print(f"Genre Names: {set(all_genre_names)}")

# %% [markdown]
# Split Dataset into a Training set and Testing set

# %%
# Split the data into training and testing sets
train_df, test_df = train_test_split(df_movie_filter, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")

# %% [markdown]
# Select the User we want to Recommend Movies to

# %%
user_id = 508

# %% [markdown]
# Define a Function for the Pre-processing technique known as "Stemming"

# %%
def stem_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed)

# %% [markdown]
# 1) Initialize Methods
# 2) Transform and Fit on Training Data
# 3) Transform and Predict on Testing Data

# %%
lda = LatentDirichletAllocation(n_components=unique_genre_count,n_jobs=-1, random_state=42) # set the Number of Topics to the Number of genres

# Define the 2 different Text Vectorization Techniques
## Remove English Stop Words & any words that show up too often or too few times
tfidf = TfidfVectorizer(norm='l2',stop_words='english', max_df=0.1, min_df =.001)
bow = CountVectorizer(stop_words='english', max_df=0.1, min_df=.001)

# Define Column Transformers
ct_tf = ColumnTransformer(
    transformers=[
        ('tfidf', tfidf, 'overview')  # Apply TF-IDF to the 'overview' column
    ]
)
ct_bow = ColumnTransformer(
    transformers=[
        ('bow', bow, 'overview')  # Apply BOW to the 'overview' column
    ]
)

# Create Pipelines to apply both Text Vectorization and LDA to the input data
pipe_tf = Pipeline(steps=[
    ('ct', ct_tf),
    ('lda', lda)
])

pipe_bow = Pipeline(steps=[
    ('ct', ct_bow),
    ('lda', lda)
])

# apply Stemming
train_df['overview_raw'] = train_df['overview'] # Create new column to apply stemming so we don't lose the input overviews


train_df['overview'] = train_df['overview'].apply(stem_text) # Apply stemming to each row of train_df['overview']

# Fit the model on the training data
pipe_tf.fit(train_df[['overview']].drop_duplicates())
pipe_bow.fit(train_df[['overview']].drop_duplicates())

# Transform the test data to get topic distributions
test_topic_distributions= pipe_tf.transform(test_df[['overview']])

# Extract the Transformers so we can see some of the outputs
tfidf_step = pipe_tf.named_steps['ct'].transformers_[0][1]
bow_step = pipe_bow.named_steps['ct'].transformers_[0][1]

# Get the feature names (words in the corpus)
feature_names_tf = tfidf_step.get_feature_names_out()
feature_names_bow = bow_step.get_feature_names_out()

# Show the Number of Dimensions for our word corpus
print(len(feature_names_tf))
print(len(feature_names_bow))

# %% [markdown]
# We want to learn a little more about how TF-IDF and BOW affect the same corpus.

# %%
# Get summary tf-idf scores for every word in the corpus
tfidf_matrix = tfidf_step.transform(train_df['overview'])
tfidf_frequencies = tfidf_matrix.sum(axis=0).A1  # Convert to 1D array
tfidf_word_freq = dict(zip(feature_names_tf, tfidf_frequencies))


# Get the 20 most common words
top_words = sorted(tfidf_word_freq.items(), key=lambda x: x[1], reverse=True)[:20] # Order words by their tfidf scores
words, freqs = zip(*top_words)

plt.figure(figsize=(12, 6))
plt.bar(words, freqs)
plt.xticks(rotation=45, ha='right')
plt.title('Top 20 Most Common Words (TF-IDF)')
plt.xlabel('Word')
plt.ylabel('TF-IDF Frequency')
plt.tight_layout()
plt.show()

# Similarly, for Bag-of-Words
bow_matrix = bow_step.transform(train_df['overview'])
bow_frequencies = bow_matrix.sum(axis=0).A1
bow_word_freq = dict(zip(feature_names_bow, bow_frequencies))
# Get the 20 most common words
top_words = sorted(bow_word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
words, freqs = zip(*top_words)

plt.figure(figsize=(12, 6))
plt.bar(words, freqs)
plt.xticks(rotation=45, ha='right')
plt.title('Top 20 Most Common Words (BOW)')
plt.xlabel('Word')
plt.ylabel('Word Frequency')
plt.tight_layout()
plt.show()

# Example
print(f"'woman tf-idf score': {tfidf_word_freq['woman']}")

idx = tfidf_step.vocabulary_['woman'] # Get index of "woman" in the vocabulary of the corpus
idf_score = tfidf_step.idf_[idx] # Get Inverse Document Frequency
print(f"IDF score for 'woman': {idf_score}")

print(f"'woman BOW score': {bow_word_freq['woman']}")


# %% [markdown]
# Define Function for Weighting the Distributions

# %%
def weighting(row):
    return [x * row['rating'] for x in row['topic_distribution']]

# %% [markdown]
# 1) Extract topic Distributions from Nested List
# 2) Apply weighting
# 3) Get mean distribution for each user

# %%
# Assign Topic Distributions into the dataset, then weight each distribution by the rating the user gave the movie
test_df['topic_distribution'] = test_topic_distributions.tolist()
display(test_df.loc[test_df['userId'] == user_id,['userId','title','topic_distribution']])

test_df['dominant_topic'] = test_df['topic_distribution'].apply(lambda x: np.argmax(x)) # Just for my knowledge, create column of the "Most Likely" Topic
test_df['weighted_topic_distribution'] = test_df.apply(weighting, axis=1) # Apply Weighting

# Assigning each Topic its own column to make data manipulations easier
topic_dist_cols = pd.DataFrame(test_df['weighted_topic_distribution'].tolist(),
                                index=test_df.index).add_prefix('weighted_topic_') # Create weighted distribution columns

test_df = pd.concat([test_df, topic_dist_cols], axis=1) # add columns to the test DataFrame


# Group by userId, calculate the mean for each weighted_topic column, create new columns in DataFrame
user_topic_means = test_df.groupby('userId')[topic_dist_cols].mean().add_prefix('average_')

# Merge the averages back into test_df
test_df = test_df.merge(user_topic_means, left_on='userId', right_index=True, how='left')

# Get all the average weighted columns
avg_topic_cols = [col for col in test_df.columns if col.startswith('average_weighted_topic_')]

user_distribution_df = test_df[['userId'] + avg_topic_cols].drop_duplicates() # Only keep one row for each User
user_distribution_df[avg_topic_cols] = normalize(user_distribution_df[avg_topic_cols],norm='l1') # Need the User's Likelihood Distributions to sum to 1
print(user_distribution_df[avg_topic_cols].sum(axis=1)) # Check Normalization works as expected

# Transform weighted columns back into a single column
user_distribution_df['average_weighted_topic_list'] = user_distribution_df[avg_topic_cols].values.tolist()


# Display the first few rows of the test DataFrame with topic distributions
display(test_df[['userId','movieId','title', 'overview','genres', 'topic_distribution', 'dominant_topic']].head(100))

# %% [markdown]
# Let's examine User 508's topic distributions for each movie seen

# %%
# Plot the topic distributions for each title for userId = 508

user_topic_df = test_df.loc[test_df['userId'] == user_id, ['title', 'topic_distribution']].copy()

plt.figure(figsize=(14, 6))

for idx, row in user_topic_df.iterrows():
    plt.plot(row['topic_distribution'], label=row['title'])
    
plt.xticks(range(0,20))
plt.xlabel('Topic Index')
plt.ylabel('Topic Probability')
plt.title(f'Topic Distributions of Each Movie seen by User {user_id}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
plt.tight_layout()
plt.show()

# %% [markdown]
# Now, let's look at the weighted distribution (before normalization)

# %%


user_topic_df = test_df.loc[test_df['userId'] == user_id, ['title', 'weighted_topic_distribution']].copy()

plt.figure(figsize=(14, 6))
for idx, row in user_topic_df.iterrows():
    plt.plot(row['weighted_topic_distribution'], label=row['title'])
    
plt.xticks(range(0,20))
plt.xlabel('Topic Index')
plt.ylabel('Topic Probability')
plt.title(f'Weighted Topic Distributions of Each Movie seen by User {user_id}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
plt.tight_layout()
plt.show()

# %% [markdown]
# Now, the normalized mean distribution for User 508

# %%

user_topic_df = user_distribution_df.loc[user_distribution_df['userId'] == user_id, ['userId','average_weighted_topic_list']].copy()

plt.figure(figsize=(14, 6))
plt.plot(user_topic_df['average_weighted_topic_list'].iloc[0], label='508')
plt.xticks(range(0,20))
plt.xlabel('Topic Index')
plt.ylabel('Topic Probability')
plt.title(f'Mean Topic Distribution for User {user_id}')
plt.tight_layout()
plt.show()

# %% [markdown]
# Before finding a similar user, let me get a sense for the movies and ratings for User 508 so I can validate any recommendations we make

# %%
# Show the counts for the values within genres in test_df for userId = 508 and show the average rating for each genre

user_rows = test_df.loc[test_df['userId'] == user_id, ['genres', 'rating']] # Get the rows in the dataFrame for User 508

# Initialize variables to hold our genre counts and ratings
genre_counts = Counter()
genre_ratings = defaultdict(list)


for idx, row in user_rows.iterrows():
    genres = row['genres']
    rating = row['rating']
    if genres:
            genres_list = ast.literal_eval(genres) # Ensure the data in the row['genres'] is a list
            for genre in genres_list:
                genre_name = genre['name'] # get Genre in list
                genre_counts[genre_name] += 1 # add 1 to the count
                genre_ratings[genre_name].append(rating) # add rating
    else:
         continue

# Calculate average rating for each genre
genre_avg_rating = {genre: (sum(ratings) / len(ratings)) for genre, ratings in genre_ratings.items()}

print("Genre counts for userId 508 in test_df:")
print(dict(genre_counts))

sorted_genre_avg_rating = dict(sorted(genre_avg_rating.items(), key=lambda x: x[1], reverse=True)) # Sort by the highest rated genres

print("Average rating for each genre for userId 508 in test_df (descending order):")
print(sorted_genre_avg_rating)

# %% [markdown]
# Define Function for Sum Squared Error (used to evaluate user distribution similarity)

# %%
def sum_squared_error(row):
    return np.sum((np.array(row['average_weighted_topic_list']) - np.array(user_508_vector)) ** 2)

# %% [markdown]
# Recommend 3 movies from the "Most Similar" Users:
# 1) Use Nearest Neighbors to find similar users (varying on distance metric)
# 2) Find 3 movies that User 508 has not seen from the "most similar" user

# %%

user_df = user_distribution_df[['userId', 'average_weighted_topic_list']].drop_duplicates(subset=['userId']) # Double Check for no duplicate users
user508_topic_df = test_df.loc[test_df['userId'] == user_id, ['title', 'weighted_topic_distribution']].copy() # Get the row for User 508

metrics = ['manhattan','cosine','minkowski'] # distance metrics

# Find the "Most Similar" user mean topic distrbutions for each distance metric
for met in metrics:

    nearest_user = NearestNeighbors(n_neighbors=5, metric=met, n_jobs=-1) # Initialize the NearestNeighbors model

    nearest_user.fit(user_df['average_weighted_topic_list'].tolist(), user_df['userId'].tolist()) # Fit the model with all users and their topic distributions
    
    distances, indices = nearest_user.kneighbors([user_df.loc[user_df['userId'] == user_id, 'average_weighted_topic_list'].values[0]]) # Find the nearest neighbors for the specified user

    nearest_user_ids = user_df.iloc[indices[0]]['userId'].values[0:3] # Get the user IDs of the nearest neighbors
    print(f"Nearest neighbors for user {user_id}: {nearest_user_ids}")

    subset = user_df[user_df['userId'].isin(nearest_user_ids)] # Get the DataFrame values for each of the 3 users

    # Plot the user distributions with our user of interest
    plt.figure(figsize=(12, 6))

    for idx, row in subset.iterrows():
        plt.plot(row['average_weighted_topic_list'], label=f"userId = {row['userId']}")

    plt.xticks(range(0,20))
    plt.xlabel('Topic Index')
    plt.ylabel('Average Weighted Topic Value')
    plt.title(f'Average Weighted Topic Distribution for Similar User ({met} distance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.tight_layout()
    plt.show()

    # Get the SSE between our user of interest and the similar users
    user_508_vector = user_df[user_df['userId'] == user_id]['average_weighted_topic_list'].values[0]

    user_df['sse_vs_508'] = user_df.apply(
        lambda row: sum_squared_error(row) if row['userId'] != user_id and row['userId'] in nearest_user_ids else np.nan, axis=1
    )
    print(user_df[['userId', 'sse_vs_508']].dropna())

    min_sse_user = user_df[user_df['userId'] != user_id].loc[user_df['sse_vs_508'].idxmin(), 'userId'] # What user has the smallest SSE?

    # Recommend Movies
    titles_508 = set(test_df[test_df['userId'] == user_id]['title']) # Get titles already seen by user 508

    user_min_sse_df = test_df[test_df['userId'] == min_sse_user] # Get the min SSE user's seen movies
    user_min_sse_df = user_min_sse_df[~user_min_sse_df['title'].isin(titles_508)] # Get the highest rated title for the minimal SSE user that user 508 hasn't seen

    # Return the 3 Highest Rated movies for User 508
    if not user_min_sse_df.empty:
        top_titles = user_min_sse_df.sort_values('rating', ascending=False).head(3)['title'].tolist()
        top_ratings = user_min_sse_df.sort_values('rating', ascending=False).head(3)['rating'].tolist()
        print(f"Recommended title for user 508: {top_titles}, Rated {top_ratings} by User {min_sse_user}")
    else:
        print("No unseen titles to recommend from the minimal SSE user.")

    # Plot the movie dsitributions for the user of interest and the "Most Similar" user
    user_topic_df = test_df.loc[test_df['userId'] == min_sse_user, ['title', 'weighted_topic_distribution']].copy()
    
    plt.figure(figsize=(14, 6))

    for idx, row in user508_topic_df.iterrows():
        plt.plot(row['weighted_topic_distribution'], label=row['title'])

    plt.xticks(range(0,20))
    plt.xlabel('Topic Index')
    plt.ylabel('Topic Probability')
    plt.title(f'Weighted Topic Distributions of Each Movie seen by User {user_id}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(14, 6))

    for idx, row in user_topic_df.iterrows():
        plt.plot(row['weighted_topic_distribution'], label=row['title'])
        
    plt.xticks(range(0,20))
    plt.xlabel('Topic Index')
    plt.ylabel('Topic Probability')
    plt.title(f'Weighted Topic Distributions of Each Movie seen by User {min_sse_user}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.tight_layout()
    plt.show()


display(user_df[user_df['userId'].isin(nearest_user_ids)])


# %% [markdown]
# Let's scope the topic distributions to just User 508 and its most similar User

# %%
# Plot the average_weighted_topic_list values for each userId in user_df[user_df['userId'].isin(nearest_user_ids)]

subset = user_df[user_df['userId'].isin(nearest_user_ids)]

plt.figure(figsize=(12, 6))
for idx, row in subset.iterrows():
    plt.plot(row['average_weighted_topic_list'], label=f"userId = {row['userId']}")

plt.xlabel('Topic Index')
plt.ylabel('Average Weighted Topic Value')
plt.title('Average Weighted Topic Distribution for Nearest Users')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
plt.tight_layout()
plt.show()


