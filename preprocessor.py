import pandas as pd
from afinn import Afinn
import nltk
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    print("Usage: preprocessor.py filepath")
    sys.exit(1)

#Load in and drop unneccessary parameter
df = pd.read_csv(filepath)
df = df.drop('Date', axis=1)

nltk.download(["stopwords", "punkt"])
#apply afinn lexicon
afinn = Afinn()
def afinn_score(text):
      return afinn.score(text)

df['title_afinn'] = df['Title'].apply(afinn_score)
df['content_afinn'] = df['Contents'].apply(afinn_score)

df['combined_afinn'] = df['title_afinn']*0.3 + df['content_afinn']*0.7
def labelInstance(score):
    if score >= 2.535:
        return 1
    elif score <= -0.27:
        return -1
    else:
        return 0

df['sentiment_class'] = [labelInstance(a) for a in df['combined_afinn']]

#Tokenize
def tokenize(text):
    tokenized_text: list[str] = nltk.word_tokenize(text)
    alpha_text = [w.lower() for w in tokenized_text if w.isalpha()]
    return alpha_text

df1 = df.copy()
def combine_columns(row):
    return row['Title'] + row['Contents']

df1['Tokens'] = df1.apply(combine_columns, axis=1)
df1['Tokens'] = df1['Tokens'].apply(tokenize)

#Clean tokens
stopwords = nltk.corpus.stopwords.words("english")
def remove_stopwords(words):
    new_words = [w for w in words if w not in stopwords]
    return new_words

df1['Tokens'] = df1['Tokens'].apply(remove_stopwords)

#Apply TF-IDF vectorization
vector_df = df1.copy()
def dummy_tokenizer(text):
    # pre-tokenization has already been done manually
    return text

#build vectorizer
#tfidf_vectorizer = TfidfVectorizer(max_features=1000, tokenizer=dummy_tokenizer, lowercase=False, ngram_range=(1, 2))
#contents_tfidf = tfidf_vectorizer.fit_transform(vector_df['Tokens'])
#joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

#load vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
contents_tfidf = tfidf_vectorizer.transform(vector_df['Tokens'])

feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(contents_tfidf.toarray(), columns=feature_names)


#Combine new features with other attributes
final_df = tfidf_df.copy()
for column in df1.columns:
    if column not in ["Title", "Contents", "Tokens", 'Label', 'combined_afinn', 'content_afinn', 'title_afinn']:
        final_df[column] = df1[column]
     
final_df.to_csv(f'preprocessed_{filepath}', index=False)
