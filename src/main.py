from tqdm import tqdm
import os
import re
import pandas
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def makeDict(data, split = True):
    """ Function to create dictionaries from the dataset
    data - the dataset to use
    split - tells the function if they split the strings even further or not
            False -> each element from the dataset is taken as one dictionary entry
            True -> each element is even further split with a ' '

    return - returns the filled dictionary based on teh given dataset
    """
    dict = []
    idxTwrd = {}
    for word in tqdm(data):
        if split:
            for i in word.split():
                if i not in dict:
                    dict.append(i)
        else:
            if word not in dict:
                dict.append(word)
    for i in dict:
        idxTwrd[i] = dict.index(i)
    return dict, idxTwrd

def tokPad(data, 
           maxPad = None, 
           removables = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
           dict = None,
           idxTwrd = None):
    """ Function to tokenize the given data
    data - Data to be tokenized
    maxPad - maximum length for padding
    dict - dictionary
    idxTwrd - reversed dictionary for tokenindexes

    """
    tokData = []
    for datum in tqdm(data):
        token = []
        pattern = '[' + ''.join(list(removables)) + ']'
        cleanData = re.sub(pattern, '', datum)
        for word in datum.split():
            token.append(idxTwrd[word])
        tokData.append(token)
    maxLen = maxPad
    padData = []
    for token in tokData:
        pad = [0] * (maxLen - len(token))
        padData.append(token + pad)
    return padData, maxLen

def oneHot(data,
           dict = None,
           idxTwrd = None):
    """ Function to create the onehot representation of the given data
    data - the vector to convert

    return - the converted array in form of a onehot vector
    """
    place = []
    for line in data:
        idx = dict.index(line)
        place.append(idx)
    a = np.array(place)
    oneHot = np.zeros((a.size, a.max()+1))
    oneHot[np.arange(a.size),a] = 1
    return oneHot

def preprocess(data, maxPad):
    """ Function to prepare the dataset

    return - X and y vectors with the right encodings and modifications
    """
    X = data['Job_offer'].to_numpy()
    y = data['Label'].to_numpy()
    Xdict, XIdxTwrd = makeDict(X)
    ydict, yIdxTwrd = makeDict(y, False)

    X, maxLen = tokPad(X, maxPad = maxPad, dict = Xdict, idxTwrd = XIdxTwrd)
    y = oneHot(y, dict = ydict, idxTwrd = yIdxTwrd)
    X = np.array(X)
    y = np.array(y)
    # X = np.expand_dims(X, 2)
    # y = np.expand_dims(y, 2)
    return X, y, Xdict, XIdxTwrd, ydict, yIdxTwrd, maxLen

def makeEmbMatrix(embedding, dict, embDepth):
    embMatrix = np.zeros((len(dict), embDepth))
    for i in tqdm(range(len(dict))):
        embeddingVec = embedding.get(dict[i])
        if embeddingVec is not None:
            embMatrix[i] = embeddingVec
    return embMatrix

def submission(X, yLabel, yPred, yDict, XDict):
    submission = ['job_description', 'Label_true', 'Label_pred']
    amount = 0
    for i in range(X.shape[0]):
        submission.append([
            detokenize(X[i], XDict),
            yDict[getIndex(yLabel[i], max(yLabel[i]))], 
            yDict[getIndex(yPred[i], max(yPred[i]))]
        ])
    # print(submission)
    with open('../data/submission.csv', "w") as file:
        for i in submission:
            line = "{};{};{}\n".format(*i)
            file.write(line)

def detokenize(sentence, dict):
    detokenized = []
    print(sentence)
    for i in sentence:
        print(i)
        detokenized.append(dict[i])
    return detokenized

def getIndex(npArray, element):
    for i in range(npArray.shape[0]):
        if npArray[i] == element:
            return i
    return None

def loadGlove(path):
    embedding = {}
    with open(path, encoding = "ISO-8859-1") as file:
        rows, cols = file.readline().split(' ')
        for line in file:
            try:
                values = line.split()
                word = values[0]
                corpus = np.array(values[1:], dtype="float32")
                embedding[word] = corpus
            except:
                pass
    return embedding, int(cols)

def model(vocabSize, embDepth, embMatrix, input_length):
    # encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=input_length)
    # encoder.adapt(train_dataset.map(lambda text, label: text))
    model = keras.models.Sequential(
        [
            # encoder,
            layers.Embedding(vocabSize, 
                             embDepth, 
                             weights=[embMatrix], 
                             input_length=input_length, 
                             trainable=False),
            layers.Bidirectional(
                layers.LSTM(64)
            ),
            # layers.LSTM(128, 
            #             return_sequences=True, 
            #             input_shape=(500, 1)),
            # layers.LSTM(128, 
            #             input_shape=(500, 1)),
            layers.Dense(5, 
                         activation='relu')
        ]
    )
    return model

if __name__ == "__main__":
    train = pandas.read_csv("../data/train_set.csv", delimiter = ",")
    test = pandas.read_csv("../data/test_set.csv", delimiter = ",")
    X, y, Xdict, XIdxTwrd, ydict, yIdxTwrd, input_length = preprocess(train, 1200)
    XTest, yTest, XTestdict, XTestIdxTwrd, yTestdict, yTestIdxTwrd, Testinput_length = preprocess(test, 1200)
    embedding, embDepth = loadGlove("../data/model.txt")
    embMatrix = makeEmbMatrix(embedding, Xdict, embDepth)
    model = model(len(Xdict), embDepth, embMatrix, input_length)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[f1_m, precision_m, recall_m])
    model.summary()
    if os.path.isfile("../data/weights.hdf5"):
        print("Weights loaded!!")
        model.load_weights("../data/weights.hdf5")
    # history = model.fit(x=X, y=y)
    # history = model.fit(x=X, y=y, validation_split = 0.2, epochs = 8)
    history = model.fit(x=X, y=y, validation_split = 0.3, steps_per_epoch = 500)
    model.save_weights("../data/weights.hdf5")
    print(history.params)
    out = model.predict(XTest)
    print(ydict)
    submission(XTest, yTest, out, ydict, Xdict)
