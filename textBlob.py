import pandas as pd
from textblob import TextBlob


def spell_correct(file_path='data/train.csv'):
    rats_tr = pd.read_csv(file_path)
    comments = rats_tr.comments.fillna('').tolist()
    new_comments = []
    for i in xrange(len(comments)):
        print i
        new_comments.append(TextBlob(comments[i]).correct().raw)
    rats_tr['comments'] = pd.Series(comments)
    rats_tr.to_csv(file_path.replace('.csv', '_correct.csv'))


def gen_sentiment(file_path='data/train.csv'):
    rats_tr = pd.read_csv(file_path)
    comments = rats_tr.comments.fillna('').tolist()
    polarity = []
    subjectivity = []
    for i in xrange(len(comments)):
        print i
        t = TextBlob(comments[i])
        polarity.append(t.sentiment.polarity)
        subjectivity.append(t.sentiment.subjectivity)
    rats_tr['polarity'] = pd.Series(polarity)
    rats_tr['subjectivity'] = pd.Series(subjectivity)
    rats_tr.to_csv(file_path.replace('.csv', '_senti.csv'))


spell_correct()