from sklearn.model_selection import train_test_split
# To count the number of text vectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# feature extraction - TF-IDF 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
# for sequential model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, CuDNNLSTM
from keras.optimizers import SGD, Adadelta
# for saving countvectorizer
import pickle


review = pd.read_csv(r'chat.csv')

print(review)

'''review['Text'] = ["Hello, hi, hola", "goodbye, bye, see you later"]
review['class'] = ["greet", "goodbye"]'''
 
# Split the data into train & test
# X_train, X_test, y_train, y_test = train_test_split(review['Text'], review['class'], random_state = 0)

X_train = review['Text']
y_train = review['class']

# min_df = 5, ngram_range = (1,3)

vect = CountVectorizer().fit(X_train)
print(vect)

feat = vect.get_feature_names()
'''cloud = WordCloud(width=1440, height=1080).generate(" ".join(feat))
# larger the size of the word, more the times it appears
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()'''

X_train_vectorized = vect.transform(X_train)

print(X_train_vectorized)

print("")

print(X_train_vectorized.shape[1])

# logistic model
'''model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# accuracy :- 
predictions = model.predict(vect.transform(X_test))
print(accuracy_score(y_test, predictions))'''

# sequential
model = Sequential()
'''model.add(CuDNNLSTM(50, input_shape=(len(train_x[0])), return_sequences = True))
model.add(CuDNNLSTM(128))'''
# X_train_vectorized.shape[0]

model.add(Dense(128, input_shape=(X_train_vectorized.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='softmax'))
# y_train.shape[0]
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adadelta(), metrics=['sparse_categorical_accuracy'])
# model summary
print(model.summary())

# fitting and saving the model 
hist = model.fit(X_train_vectorized, y_train, epochs=50, batch_size=5, verbose=1)
# hist = model.fit(x, y, epochs=200, batch_size=5, verbose=1)
model.save('vec_model.h5', hist)
print("model created")

pickle.dump(vect, open("vect.pickle", "wb"))
print("vect saved")

msg = "I have 5 days of holidays"
msg = vect.transform(msg)

print(model.predict(msg))