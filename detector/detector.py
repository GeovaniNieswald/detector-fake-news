import pandas as pd
import re, os
import multiprocessing
import numpy as np
import datetime as dt
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

# TOTAL DE NOTÍCIAS (34.900)
#  FAKE (17.450)
#  REAL (17.450)

# Utilizado para controlar se serão adicionadas palavras semelhantes 
# a sentença que será classificada, utilizando para isso WE (Word2Vec)
utilizar_we = False

# Local onde o modelo de rede neural será salvo
lstm_model_filename = 'model/lstm.h5'

# Local onde os logs de treinamento serão salvos
logs_fit_dir = 'logs/fit/com_we' if utilizar_we == True else 'logs/fit/sem_we'

# Local onde os resultados serão salvos
resultados_filename = 'logs/output/com_we/resultados.txt' if utilizar_we == True else 'logs/output/sem_we/resultados.txt'

# Quantidade máxima de palavras no vocabulário
max_words = 10000

# Tamanho máximo de todas as sentenças
max_sequence_length = 2000

# Dimensão de saída da camada Embedding
embed_dim = 128

# Quantidade de vezes que o dataset passará pela rede neural
epochs = 5

# Número de amostras utilizadas em cada atualização do gradiente
batch_size = 248

# Normalizar dados com NLTK
def normalizar(dados):
    # remove colunas desnecessárias
    try:
        dados = dados.drop("Unnamed: 0", axis = 1)
        dados = dados.drop("Unnamed: 0.1", axis = 1)        
    except KeyError:
        pass
    
    # remove linhas que tenham alguma coluna em branco
    dados.replace("", float("NaN"), inplace=True)
    dados.dropna(subset = ["title", "text", "label"], inplace=True)

    # colocar tudo em minusculo
    dados["title"] = dados["title"].str.lower()
    dados["text"]  = dados["text"].str.lower()

    # remove pontuação e caracteres especiais
    dados['title'] = dados['title'].apply((lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)))
    dados['text']  = dados['text'].apply((lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)))

    # remove stopwords (palavras vazias)
    palavras_vazias = set(stopwords.words('english')) 
    for i, texto in enumerate(dados['text'].values):
        palavras = word_tokenize(texto)
        palavras_sem_palavras_vazias = [palavra for palavra in palavras if not palavra in palavras_vazias]
        dados.iat[i, 1] = palavras_sem_palavras_vazias

    return dados

# Pré-processar dados (tokenização, divisão de treino/teste)
def pre_processar(dados):
    text = dados['text'].values

    if utilizar_we:
        text = utilizar_word_embedding(text)

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(text)

    # transforma o texto - string em int
    text_sequences = tokenizer.texts_to_sequences(text)  
    text_sequences = pad_sequences(text_sequences, maxlen=max_sequence_length)

    text_labels = pd.get_dummies(dados['label']).values

    # separa o dataset em dois conjuntos, um para treino e outro para testes
    x_train, x_test, y_train, y_test = train_test_split(text_sequences, text_labels, test_size=0.10, random_state=42)

    return x_train, x_test, y_train, y_test, tokenizer

# Word Embedding (Word2Vec) - Tenta adicionar palavras semelhantes ao texto
def utilizar_word_embedding(text):
    w2v = Word2Vec(text, window=5, min_count=5, workers=multiprocessing.cpu_count(), sg=1)

    for i, texto in enumerate(text):
        aux = []
    
        for palavra in texto:
            aux.append(palavra)
    
            try:
                similar = w2v.wv.most_similar(positive=[palavra], topn = 1)
    
                # verifica se a palvra encontrada é semelhante o suficiente e add ao texto
                if similar[0][1] >= 0.9:
                    aux.append(similar[0][0])           
            except KeyError:
                continue
    
        text[i] = aux

    return text

# Cria o modelo da rede neural LSTM com suas camadas
def criar_modelo():
    input_shape = (max_sequence_length,)
    model_input = Input(shape=input_shape, name="input", dtype='int32')    

    embedding = Embedding(max_words, embed_dim, input_length=max_sequence_length, name="embedding")(model_input)
    
    lstm = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)
    
    model_output = Dense(2, activation='softmax', name="softmax")(lstm)

    modelo = Model(inputs=model_input, outputs=model_output)
    modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return modelo

# Treina a rede neural
def treinar_rede(modelo, x_train, y_train):
    tensoarboard_callback = TensorBoard(log_dir=logs_fit_dir, histogram_freq=1)

    resultado_treinamento = modelo.fit(
        x_train, 
        y_train, 
        validation_split=0.20,
        epochs=epochs, 
        batch_size=batch_size, 
        shuffle=True, 
        verbose=1,
        callbacks=[tensoarboard_callback])
    
    return resultado_treinamento

# Realiza a análise do treinamento da rede neural
def analisar_treino(resultado_treinamento):
    loss = resultado_treinamento.history['loss']
    val_loss = resultado_treinamento.history['val_loss']

    try:
        accuracy = resultado_treinamento.history['accuracy']
    except KeyError:
        accuracy = resultado_treinamento.history['acc']
    
    try:
        val_accuracy = resultado_treinamento.history['val_accuracy']
    except KeyError:
        val_accuracy = resultado_treinamento.history['val_acc']

    arquivo = open(resultados_filename,'a')

    arquivo.write("Perda no treinamento\n")
    for i, l in enumerate(loss):
        arquivo.write("epoch " + str(i + 1) + ": %.2f%%" % (l*100) + "\n")

    arquivo.write("\n")
    arquivo.write("Perda na validação\n")
    for i, val_l in enumerate(val_loss):
        arquivo.write("epoch " + str(i + 1) + ": %.2f%%" % (val_l*100) + "\n")

    arquivo.write("\n")
    arquivo.write("Acurácia no treinamento\n")
    for i, acc in enumerate(accuracy):
        arquivo.write("epoch " + str(i + 1) + ": %.2f%%" % (acc*100) + "\n")

    arquivo.write("\n")
    arquivo.write("Acurácia na validação\n")
    for i, val_acc in enumerate(val_accuracy):
        arquivo.write("epoch " + str(i + 1) + ": %.2f%%" % (val_acc*100) + "\n")
 
    arquivo.close()

# Realiza a análise do modelo
def analisar_modelo(modelo, x_test, y_test):
    resultado_avaliacao = modelo.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
    resultado_predicao  = modelo.predict(x_test)
    
    rounded_predictions = np.argmax(resultado_predicao, axis=1)
    rounded_labels      = np.argmax(y_test, axis=1)
    tn, fp, fn, tp      = confusion_matrix(rounded_labels, rounded_predictions).ravel()

    arquivo = open(resultados_filename,'a')
    arquivo.write("Acurácia com dados de teste: %.2f%%" % (resultado_avaliacao[1]*100) + "\n\n")
    arquivo.write("Matriz de confusão\n")
    arquivo.write("tn: " + str(tn) + "\n")
    arquivo.write("fp: " + str(fp) + "\n")
    arquivo.write("fn: " + str(fn) + "\n")
    arquivo.write("tp: " + str(tp) + "\n")
    arquivo.close()

# Loop para realizar a classificação de novas sentenças
def detectar_fake_news(tokenizer, modelo):
    while True:
        print("")
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

with tf.device("/gpu:0"):
    arquivo = open(resultados_filename,'w')
    arquivo.write("Resultados\n\n")
    arquivo.close()

    treinar = not os.path.exists('./{}'.format(lstm_model_filename))
    dados = pd.read_csv('./dataset/news.csv')

    dados = normalizar(dados)

    inicio = dt.datetime.now()
    x_train, x_test, y_train, y_test, tokenizer = pre_processar(dados)
    fim = dt.datetime.now()

    tempo_preprocessamento = fim - inicio

    modelo = criar_modelo()

    print("")
    print(modelo.summary())
    print("")

    if treinar:
        inicio = dt.datetime.now()
        resultado_treinamento = treinar_rede(modelo, x_train, y_train)
        fim = dt.datetime.now()

        tempo_treinamento = fim - inicio

        modelo.save_weights(lstm_model_filename)   
        analisar_treino(resultado_treinamento)
    else:
        tempo_treinamento = 0
        modelo.load_weights('./{}'.format(lstm_model_filename))

    arquivo = open(resultados_filename,'a')
    arquivo.write("Tempo de Pré-processamento\n")
    arquivo.write(str(tempo_preprocessamento) + "\n\n")
    arquivo.write("Tempo de Treinamento\n")
    arquivo.write(str(tempo_treinamento) + "\n\n")
    arquivo.write("Tempo de Pré-processamento + Tempo de Treinamento\n")
    arquivo.write(str(tempo_preprocessamento + tempo_treinamento) + "\n")
    arquivo.close()

    analisar_modelo(modelo, x_test, y_test)

    detectar_fake_news(tokenizer, modelo)