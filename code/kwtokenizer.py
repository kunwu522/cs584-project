import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))


def sentence_tokenize(text, no_stopwords=False, stem=False):
    # remove URL link
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]\
        |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '',
        text
    )
    text = re.sub(r'#(\w+)', '', text)  # remove hashtag
    text = re.sub(r'([.!.?])', r'\1 ', text)  # Add a space after .!?
    text = re.sub(r'\s+', ' ', text)

    sent_tokens = []
    sentences = sent_tokenize(text)
    for sent in sentences:
        sent = sent.strip().lower()
        sent = re.sub(r'[^a-zA-Z\s]', '', sent)
        tokens = word_tokenize(sent)

        if no_stopwords:
            tokens = [token for token in tokens if token not in stop_words]

        if stem:
            tokens = [stemmer.stem(token) for token in tokens]

        if len(tokens) == 0:
            continue

        sent_tokens.append(tokens)
    return sent_tokens


def tokenize(text, no_stopwords=False, stem=False):
    tokens = []
    sent_tokens = sentence_tokenize(text, no_stopwords, stem)
    for st in sent_tokens:
        tokens.extend(st)
    return tokens


if __name__ == "__main__":
    text = "This is so cool. It's like, 'would you want your \
            mother to read this??' Really great idea, well done!"
    print(tokenize(text))
