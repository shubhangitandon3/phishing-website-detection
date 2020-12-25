import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from keras.models import Sequential,load_model
from keras.layers import Dense, Activation ,Dropout , Flatten , Conv1D ,MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
from timeit import default_timer as timer
import inputScript

def CNNLSTMbuild_model(input):
	model = Sequential()
	model.add(Dense(128,input_shape=(input[1],input[2])))
	model.add(Conv1D(filters = 24, kernel_size= 1,padding='valid', activation='relu', kernel_initializer="uniform"))
	model.add(MaxPooling1D(pool_size=2, padding='valid'))
	model.add(Conv1D(filters = 48,kernel_size = 1,padding='valid', activation='relu', kernel_initializer="uniform"))
	model.add(MaxPooling1D(pool_size=2, padding='valid'))
	model.add(LSTM(40,return_sequences=True))
	model.add(LSTM(32,return_sequences=False))
	model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
	#model.add(Dropout(0.2))
	model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
	model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	return model

def process(path,inputurl):
	dataset = pd.read_csv(path)
	
	x = dataset.iloc[ : , :-1].values
	y = dataset.iloc[:, -1:].values
	
	#spliting the dataset into training set and test set
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )

	classifier = XGBClassifier(random_state = 0)
	classifier.fit(x_train, y_train)
	checkprediction = inputScript.process(inputurl)
	x_test1 = np.array(checkprediction)
	print("x_test1=",x_test1)

	#predicting the tests set result
	y_pred = classifier.predict(x_test1)
	print("prediction=",y_pred)
	return y_pred

##	model = load_model('results/CNNLSTM.h5')
##	df = pd.read_csv(path)
##	
##	x = df.iloc[ : , :-1].values
##	y = df.iloc[:, -1:].values
##	
##	classifier = CNNLSTMbuild_model([x.shape[0], x.shape[1],1])
##
##
##	#test_X=[]
##
##	checkprediction = inputScript.process(inputurl)
##	x_test = np.array(checkprediction)
##	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1)) 
##	print("x_text",x_test)
##	print(x_test.shape)
##	prediction = classifier.predict(x_test)
##	print("prediction",prediction[0][0])
##	result=""
##	if prediction[0][0] > 0:
##		result=1
##	else:
##		result=-1
##	print(result)
#	return result

#process("data.csv","http://ssstrades.com/Chase/chase")

