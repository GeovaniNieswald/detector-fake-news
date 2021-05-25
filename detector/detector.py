import pandas as pd
import re, os
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

# TOTAL DE NOTÍCIAS (6335)
#  FAKE (3164)
#  REAL (3171)

# DADOS TREINAMENTO (3168)
#  FAKE (1582)
#  REAL (1586)

# DADOS DE DETECÇÃO (3167)
#  FAKE (1582)
#  REAL (1585)

utilizar_we = False
lstm_model_filename = 'model/lstm.h5'
max_fatures = 5000
max_sequence_length = 300
embed_dim = 128
epochs = 5
batch_size = 32

# Normalizar dados com NLTK
def normalizar(dados):
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
    dados['title'] = dados['title'].apply((lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)))
    dados['text'] = dados['text'].apply((lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)))

    # remove stopwords (palavras vazias)
    palavras_vazias = set(stopwords.words('english')) 
    for i, texto in enumerate(dados['text'].values):
        palavras = word_tokenize(texto)
        palavras_sem_palavras_vazias = [palavra for palavra in palavras if not palavra in palavras_vazias]
        dados.iat[i, 1] = palavras_sem_palavras_vazias

    return dados

# Pré-processamento
def pre_processamento(dados):
    text = dados['text'].values

    if utilizar_we:
        text = utilizar_word_embedding(text)

    tokenizer = Tokenizer(num_words=max_fatures)
    tokenizer.fit_on_texts(text)

    text_sequences = tokenizer.texts_to_sequences(text)  
    text_sequences = pad_sequences(text_sequences, maxlen=max_sequence_length)

    text_labels = pd.get_dummies(dados['label']).values

    x_train, x_test, y_train, y_test = train_test_split(text_sequences, text_labels, test_size=0.20, random_state=42)

    return x_train, x_test, y_train, y_test, tokenizer

# Word Embedding (Word2Vec)
def utilizar_word_embedding(text):
    w2v = Word2Vec(text, window=5, min_count=5, negative=15, workers=multiprocessing.cpu_count())

    for i, texto in enumerate(text):
        aux = []
    
        for palavra in texto:
            aux.append(palavra)
    
            try:
                similar = w2v.wv.most_similar(positive=[palavra], topn = 1)
    
                if similar[0][1] >= 0.9:
                    aux.append(similar[0][0])           
            except KeyError:
                continue
    
        text[i] = aux

    return text

# Cria a rede neural LSTM
def criar_modelo():
    input_shape = (max_sequence_length,)
    model_input = Input(shape=input_shape, name="input", dtype='int32')    

    embedding = Embedding(max_fatures, embed_dim, input_length=max_sequence_length, name="embedding")(model_input)
    
    lstm = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)
    
    model_output = Dense(2, activation='softmax', name="softmax")(lstm)

    modelo = Model(inputs=model_input, outputs=model_output)
    modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return modelo

# Treina a rede neural
def treinar_rede(modelo, x_train, y_train, x_test, y_test):
    resultado_treinamento = modelo.fit(
        x_train, 
        y_train, 
        validation_data=(x_test, y_test), 
        epochs=epochs, 
        batch_size=batch_size, 
        shuffle=True, 
        verbose=1)
    
    return resultado_treinamento

# Exibe gráficos para análise
def analisar_modelo(modelo, resultado_treinamento, x_test, y_test):
    print(modelo.summary())

    plt.figure()
    plt.plot(resultado_treinamento.history['loss'], lw=2.0, color='b', label='train')
    plt.plot(resultado_treinamento.history['val_loss'], lw=2.0, color='r', label='val')
    plt.title('Detecção de Fake News')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure()
    plt.plot(resultado_treinamento.history['accuracy'], lw=2.0, color='b', label='train')
    plt.plot(resultado_treinamento.history['val_accuracy'], lw=2.0, color='r', label='val')
    plt.title('Detecção de Fake News')
    plt.xlabel('Epochs')
    plt.ylabel('Acurácia')
    plt.legend(loc='upper left')
    plt.show()

    scores = modelo.evaluate(x_test, y_test, verbose = 0, batch_size = batch_size)
    print("Acc: %.2f%%" % (scores[1]*100))

# Loop para realizar a detecção de FN
def detectar_fake_news(tokenizer, modelo):
    while True:
        sentence = input("input> ")

        if sentence == "exit":
            break
        
        new_text = [sentence]
        new_text = tokenizer.texts_to_sequences(new_text)
        new_text = pad_sequences(new_text, maxlen=max_sequence_length, dtype='int32', value=0)

        analise = modelo.predict(new_text, batch_size=1, verbose=2)[0]

        if(np.argmax(analise) == 0):
            pred_proba = "%.2f%%" % (analise[0] * 100)
            print("Falso => ", pred_proba)
        elif (np.argmax(analise) == 1):
            pred_proba = "%.2f%%" % (analise[1] * 100)
            print("Verdadeiro => ", pred_proba)

# -------------------------------------------------------------------------

treinar = not os.path.exists('./{}'.format(lstm_model_filename))
dados = pd.read_csv('./dataset/news_treinamento_p.csv')

dados = normalizar(dados)

x_train, x_test, y_train, y_test, tokenizer = pre_processamento(dados)

modelo = criar_modelo()

if treinar:
    resultado_treinamento = treinar_rede(modelo, x_train, y_train, x_test, y_test)
    modelo.save_weights(lstm_model_filename)   
    analisar_modelo(modelo, resultado_treinamento, x_test, y_test)
else:
    modelo.load_weights('./{}'.format(lstm_model_filename))

detectar_fake_news(tokenizer, modelo)