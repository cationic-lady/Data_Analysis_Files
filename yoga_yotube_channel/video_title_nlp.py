import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Function to clean up text and remove any special characters
def clean_text(content):
    '''
    clean the text by removing 
    special characters with regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", content).split())

# Function to lemmatize and preprocess text
def preprocess_title(title):
    tokens = title.lower().split()  # Tokenize and convert to lowercase
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalnum()]  # Lemmatize and remove stopwords
    return ' '.join(tokens)


# Read the raw dataset
df = pd.read_csv("youtube_yoga.csv")

# Assuming the missing comment counts mean that there are no comments on those videos
df['commentCount'] = df['commentCount'].fillna(0)

# Engagement Ratio = (comments+likes)/views column 
df['engagement_ratio'] = (df['likeCount']+df['commentCount'])/df['viewCount']

# Rearrange the dataset in a chronological order (optional)
df = df.sort_values('release_date', ascending=True)
df.reset_index(drop=True, inplace=True)

# Consider only those videos with >1 views
df = df[df['viewCount']>0].reset_index(drop=True)

# Our data is preprocessed; onto the NLP
df_title = df[['videoTitle','engagement_ratio']]

# Clean the titles using clean_title(name) function
df_title['clean_titles'] = [clean_text(title) for title in df_title['videoTitle']] 

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess the titles
df_title['lemmatized_titles'] = df_title['clean_titles'].apply(preprocess_title)

# Tokenize and preprocess titles using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')  # Automatically removes stopwords
X = vectorizer.fit_transform(df_title['lemmatized_titles'])  # Create word frequency matrix
keywords = vectorizer.get_feature_names_out()  # Extract unique tokens (keywords)

# Convert word matrix to DataFrame
keywords_df = pd.DataFrame(X.toarray(), columns=keywords)

# Combine with the original DataFrame
df_keywords = pd.concat([df_title, keywords_df], axis=1)

# Calculate engagement ratio metrics for each keyword
keyword_impact = {}

for keyword in keywords:
    count_engagement = df_keywords[df_keywords[keyword] == 1]['engagement_ratio'].count()
    mean_engagement = df_keywords[df_keywords[keyword] == 1]['engagement_ratio'].mean()
    std_engagement = df_keywords[df_keywords[keyword] == 1]['engagement_ratio'].std()
    keyword_impact[keyword] = [count_engagement, mean_engagement, std_engagement]

# Convert to DataFrame for visualization
keyword_impact_df = pd.DataFrame.from_dict(keyword_impact, orient='index', columns=['Count', 'Mean Engagement Ratio', 'Standard Deviation'])
keyword_impact_df = keyword_impact_df.reset_index().rename(columns={'index': 'Keyword'}).sort_values(by='Count', ascending=False)
keyword_impact_df.reset_index(drop=True, inplace=True)
keyword_impact_df = keyword_impact_df[keyword_impact_df['Count']>0]

# Our dataset is ready now; we can save it 
keyword_impact_df.to_csv("title_words.csv",index=False)