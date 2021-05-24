import pandas as pd
import re
import multiprocessing

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec

teste = True

# TOTAL DE NOTÍCIAS (6335)
#  FAKE (3164)
#  REAL (3171)

# DADOS TREINAMENTO (3168)
#  FAKE (1582)
#  REAL (1586)

# DADOS DE DETECÇÃO (3167)
#  FAKE (1582)
#  REAL (1585)

# Normalizar dados com NLTK
def normalizar(dados):
    all_words = []

    # remove coluna referente ao código
    dados = dados.drop("Unnamed: 0", axis = 1)
    dados = dados.drop("Unnamed: 0.1", axis = 1)

    # remove linhas que tenham alguma coluna em branco
    dados.replace("", float("NaN"), inplace=True)
    dados.dropna(subset = ["title", "text", "label"], inplace=True)

    # colocar tudo em minusculo
    dados["title"] = dados["title"].str.lower()
    dados["text"]  = dados["text"].str.lower()

    # remove pontuação e caracteres especiais
    dados['title'] = dados['title'].apply((lambda x: re.sub('[^A-Za-z0-9 ]+', '', x)))
    dados['text'] = dados['text'].apply((lambda x: re.sub('[^A-Za-z0-9 ]+', '', x)))

    # remove stopwords e faz a "tokenização"
    stop_words = set(stopwords.words('english')) 
    for i, text in enumerate(dados['text'].values):
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in stop_words]
        dados.iat[i, 1] = tokens_without_sw

        all_words.append(tokens_without_sw)

    return dados, all_words

# Pré-processamento com Word Embedding (Word2Vec)
def pre_processamento(all_words, dados):
    if teste:
        w2v = Word2Vec(all_words, window = 5, min_count = 5, negative = 15, workers = multiprocessing.cpu_count())
        w2v.save("models/word2vec.model")
        w2v.save("models/model.bin")
    else:
        w2v = Word2Vec.load('models/model.bin')

    for i, text in enumerate(dados['text'].values):
        for j, word in enumerate(text):
            try:
                similar = w2v.wv.most_similar(positive=[word], topn = 1)

                if similar[0][1] >= 0.9:
                    text[j] = [word, similar[0][0]]
                else:
                    text[j] = [word]                
            except KeyError:
                text[j] = [word]

        dados.iat[i, 1] = text 

    return dados

# -------------------------------------------------------------------------

dados = pd.read_csv('./dataset/news_treinamento_p.csv')
dados, all_words = normalizar(dados)

dados_processados = pre_processamento(all_words, dados)

print(dados_processados)