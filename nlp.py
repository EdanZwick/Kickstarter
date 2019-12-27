from gensim.models import Word2Vec, FastText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk


def make_word_embeddings(df):
    print('started nlp')
    if os.path.isfile('./fastText.model'):
        print('started reading pretrained model file')
        model = FastText.load('./fastText.model')
        print('finished file loading')
    else:
        df1 = df[['name', 'blurb']]
        df1['name'] = df1.apply(lambda r: re.sub("[^A-Za-z']", ' ', r['name'].lower()), axis=1)
        df1['blurb'] = df1.apply(lambda r: re.sub("[^A-Za-z']", ' ', str(r['blurb']).lower()), axis=1)
        df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)

        df_clean = pd.DataFrame({'clean': df2})
        sent = [row.split(',') for row in df_clean['clean']]
        model = FastText(sentences=sent, min_count=1, size=20, window=3, iter=10)
        model.save("fastText.model")

    print('started setting name_nlp column')
    df['name_nlp'] = df.apply(lambda row: sum([model.wv[re.sub("[^A-Za-z']", ' ', s.lower().strip())] for s in
                                               (row["name"] + ' ' + str(row['blurb'])).split(' ')]), axis=1)
    print('finished setting name_nlp column')


def set_semantics(df):
    analyser = SentimentIntensityAnalyzer()
    neg = []
    pos = []
    compound = []

    for row in df['blurb'].astype(str):
        score = analyser.polarity_scores(str(row.encode("utf-8")))
        neg.append(score['neg'])
        pos.append(score['pos'])
        compound.append(score['compound'])

    df['blurb_pos'] = pos
    df['blurb_neg'] = neg
    df['blurb_compound'] = compound


def avg_word(sentence):
    words = sentence.split()
    return float(sum(len(word) for word in words)) / len(words)


def set_text_statistics(df):
    nltk.download('stopwords')
    df_str = df["blurb"].astype(str)

    stopwords = set(nltk.corpus.stopwords.words('english'))
    df["name_num_words"] = df["name"].apply(lambda x: len(x.split()))
    df["name_num_chars"] = df["name"].apply(lambda x: len(x.replace(" ", "")))
    df['name_avg_word_length'] = df['name'].apply(lambda x: avg_word(x))
    df["blurb_num_words"] = df_str.apply(lambda x: len(x.split()))
    df["blurb_num_chars"] = df_str.apply(lambda x: len(x.replace(" ", "")))
    df['blurb_avg_word_length'] = df_str.apply(lambda x: avg_word(x))
    df['blurb_stopwords'] = df_str.apply(lambda x: len([x for x in x.split() if x in stopwords]))
    df['name_upper'] = df['name'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    df['blurb_upper'] = df_str.apply(lambda x: len([x for x in x.split() if x.isupper()]))
