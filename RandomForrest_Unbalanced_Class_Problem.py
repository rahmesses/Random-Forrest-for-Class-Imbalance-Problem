# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:03:53 2020

@author: Rahmesses
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


class PredictOutcome(): 
    
    def Read_Data(): # Read data from input path
        
        train = pd.read_csv(path + "/" + "train.csv")
        test = pd.read_csv(path+ "/" + "test.csv")
        
        return train,test
    
    def Preprocess(df): # Data preprocessing
        
        df.gender =df.gender.fillna(df.gender.mode()[0]) # Fill NA with mode of gender
       
    # Normalize continuous numeric features and create dummies for categrical features
        
        normalize = preprocessing.MinMaxScaler() 
        df=pd.get_dummies(df,columns=['device_type','gender','in_initial_launch_location','n_drivers','n_vehicles'])
        df.age = normalize.fit_transform(df[['age']].astype(np.float))
        df.income = normalize.fit_transform(df[['income']].astype(np.float))
        df.cost_of_ad = normalize.fit_transform(df[['cost_of_ad']].astype(np.float))
        df.prior_ins_tenure = normalize.fit_transform(df[['prior_ins_tenure']].astype(np.float))
        
        return df
       
    def TrainModel(): #Model Training
        
        train = PredictOutcome.Read_Data()[0]
        train= PredictOutcome.Preprocess(train)
        
    # Data is imbalanced. Undersampling the class in majority and Oversampling class in minority to match the observation count
        
        outcome_0 = train[train.outcome==0]
        outcome_1 = train[train.outcome==1]
        train_downsampled_0 = resample(outcome_0,replace=True,n_samples=len(outcome_0)//5,random_state=27)
        train_upsampled_1 = resample(outcome_1,replace=True,n_samples=len(outcome_0)//5,random_state=27)
        
        train = pd.concat([train_downsampled_0,train_upsampled_1])
        
    # Creating numpy arrays for model input
        
        X = train.loc[:,train.columns!='outcome'].values
        Y = train.loc[:,train.columns=='outcome'].values
        
    # Split the training data into train and validate

        X_train, X_validate, Y_train, Y_validate = train_test_split(X,Y,test_size=0.2, random_state=25)
        
    # Random Forest Model and predictions
#        
        RF_Model = RandomForestClassifier(n_estimators = 200, random_state = 42)
        RF_Model.fit(X_train,Y_train.ravel())
        pred_RF = RF_Model.predict(X_validate)
        
#        model = Sequential()
#        model.add(Dense(1000, input_dim = 18, activation= 'sigmoid'))
#        #model.add(Dropout(0.2))
#        model.add(Dense(1000, activation= 'relu'))
#        model.add(Dropout(0.2))
#        model.add(Dense(1000, activation= 'relu'))
#        model.add(Dropout(0.2))
#        #model.add(Dense(100, activation= 'relu'))
#       # model.add(Dropout(0.2))
#        #model.add(Dense(100, activation= 'relu'))
#        #model.add(Dropout(0.2))
#        model.add(Dense(1, activation= 'sigmoid'))
#        
#        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#        
#        ## Training
#        
#        model.fit(X_train, Y_train, epochs=25, batch_size=100)
#        
#        ## Accuracy
#        
#        _, accuracy = model.evaluate(X_train, Y_train)
#        print('Accuracy: %.2f' % (accuracy*100))
#        
#        ## Predictions
#        
#        predictions = model.predict_classes(X_validate)
#        
        
        ## 10 predictions
        
        #for i in range(10):
        	#print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], Y_test[i]))
        
        
    # Model Evaluation Metrics. Plot ROC and Calculate AUC
        
        fpr_RF = dict()
        tpr_RF = dict()
        roc_auc_RF = dict()
        
        fpr_RF, tpr_RF, _ = roc_curve(Y_validate, pred_RF)
        roc_auc_RF = auc(fpr_RF,tpr_RF)
        
        plt.figure()
        lw = 2
        plt.plot(fpr_RF, tpr_RF, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc_RF)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Random Forest')
        plt.legend(loc="lower right")
        plt.show()
        
        print("AUC of the model:",auc(fpr_RF,tpr_RF))
        
        # Store Model for later use in the designated path
        
       # joblib.dump(RF_Model, path + "/" + "NN_Model")
        
    def Predict():
        
        # Preprocessing test data for model input
        
        test_original = PredictOutcome.Read_Data()[1]
        test = PredictOutcome.Preprocess(test_original)
        
        # Load stored model
        
        RF_Model = joblib.load(path + "/" + "RF_Model")
        
        predict = RF_Model.predict(test) # Predictions
        
        test_original['outcome'] = predict # Add predicted values in test_original
        
        return test_original.to_csv(path + "/" + "PredictedTest.csv",index=False) # Store data in CSV file
        
    def main():
        #global path
        #path = input("Please enter the path")
        PredictOutcome.TrainModel()
        PredictOutcome.Predict()
        
if __name__ == "__main__":
    
    path = input("Please enter the path of stored files")
    PredictOutcome.main()