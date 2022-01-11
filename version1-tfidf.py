# https://www.kaggle.com/jinpei/jigsaw-tfidf-ridge-0-81/edit


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
import scipy
pd.options.display.max_colwidth=300


#
df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
print(df.shape)

# Give more weight to severe toxic 
df['severe_toxic'] = df.severe_toxic * 2
df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
df.sample(5)

#Reduce the rows with 0 toxicity
df = pd.concat([df[df.y>0] , 
                df[df.y==0].sample(int(len(df[df.y>0])*1.5)) ], axis=0).sample(frac=1)

print(df.shape)


# Create Sklearn Pipeline with
# TFIDF - Take 'char_wb' as analyzer to capture subwords well
# Ridge - Ridge is a simple regression algorithm that will reduce overfitting


pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))),
        #("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
        ("clf", Ridge()),
        #("clf",LinearRegression())
    ]
)


vectorizer = TfidfVectorizer()
vectorizer.fit_transform(df['text'])


# Train the pipeline
pipeline.fit(df['text'], df['y'])

df_val = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")


p1 = pipeline.predict(df_val['less_toxic'])
p2 = pipeline.predict(df_val['more_toxic'])

f'Validation Accuracy is { np.round((p1 < p2).mean() * 100,2)}'
