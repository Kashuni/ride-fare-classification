
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.impute import SimpleImputer
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier as dtc


def predict():

  my_imputer = SimpleImputer()
  imputed_new_features = pd.DataFrame(my_imputer.fit_transform(new_features))
  #imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X.drop(['pickup_time','drop_time'],axis='columns')))
  imputed_new_features.columns = new_features.columns
  imputed_new_test_features = pd.DataFrame(my_imputer.transform(new_test_features))
  imputed_new_test_features.columns = new_test_features.columns

  preds_rand = pd.DataFrame(classifier.predict(imputed_new_test_features))
  preds_rand.columns = ['prediction']
  preds_rand['tripid'] = test['tripid']
  f_cols = ['tripid','prediction']
  preds_rand = preds_rand[f_cols]

  preds_rand['prediction'] = preds_rand['prediction'].astype(int)

  return preds_rand


def classify(df, classifier_algo, train_size):

  Y = norm_train.label

  x_train, x_valid, y_train, y_valid = train_test_split(df, Y, train_size=train_size, test_size=(1-train_size), random_state=0)

  if classifier_algo == 1:
    for i in range(1,50):
      classifier = KNeighborsClassifier(n_neighbors=i, p=1, leaf_size=1, weights='uniform')
      classifier.fit(x_train, y_train)
      preds = classifier.predict(x_valid)
      print("KNN -> Depth - {}, score {}".format(i, f1_score(y_valid, preds, average='macro')))

  elif classifier_algo == 2:
    for i in range(1,35):
      classifier = RandomForestClassifier(max_depth=27, random_state=102435, n_estimators=22)
      classifier.fit(x_train, y_train)
      preds = classifier.predict(x_valid)
      print("Random Forest -> Depth - {}, score {}".format(i, f1_score(y_valid, preds, average='macro')))


def normalize(imputed_X_train):

  transformer = RobustScaler().fit(imputed_X_train)

  normalized_X = pd.DataFrame(transformer.transform(imputed_X_train))
  normalized_test = pd.DataFrame(transformer.transform(test))

  normalized_X.columns = imputed_X_train.columns
  normalized_test.columns = test.columns
  
  #normalized_X.describe()
  dropping_clms = ['additional_fare','meter_waiting']
  normalized_X.drop(columns=dropping_clms,axis=1,inplace=True)
  normalized_test.drop(columns=dropping_clms,axis=1,inplace=True)

  imputed_X_train.drop('tripid',axis=1,inplace=True)
  test.drop('tripid',axis=1,inplace=True)


def distance(lt1, lt2, ln1, ln2):
  dist_array = []
  
  for i in range (len(lt1)): 
    ln1_i = radians(ln1[i]) 
    ln2_i = radians(ln2[i]) 
    lt1_i = radians(lt1[i]) 
    lt2_i = radians(lt2[i])
        
    # Haversine formula  
    dln = ln2_i - ln1_i  
    dlt = lt2_i - lt1_i 
    a = sin(dlt / 2)*2 + cos(lt1_i) * cos(lt2_i) * sin(dln / 2)*2
    c = 2 * asin(sqrt(abs(a)))  
      
    # Radius of earth in kilometers. Use 3956 for miles 
    r = 6371
        
    dist = c * r
    dist_array.append(dist)

  return dist_array


def impute(df):
  imp = SimpleImputer()
  imp_df_train = pd.DataFrame(imp.fit_transform(df))
  imp_df_train.columns = df.columns

  imp_df_train['distance'] = distance(imp_df_train['pick_lat'], imp_df_train['drop_lat'], imp_df_train['pick_lon'], imp_df_train['drop_lon'])
  
  imp_df_train.drop(['pick_lat','drop_lat','pick_lon','drop_lon'],axis=1,inplace=True)

  Y.drop(imp_df_train.loc[imp_df_train.additional_fare>5000].index,axis=0,inplace=True)
  imp_df_train.drop(imp_df_train.loc[imp_df_train.additional_fare>5000].index,axis=0,inplace=True)

  imp_df_train.drop(['duration'],axis=1,inplace=True)

  return imp_df_train


def preprocess(df, test=0):
  
  df.drop('label',axis=1,inplace=True)

  df['time_dur'] = (df.drop_time-df.pickup_time).astype('timedelta64[s]')

  df.drop(['pickup_time','drop_time'],axis='columns',inplace=True)

  df['pickup_time'] = pd.to_datetime(df['pickup_time'])
  df['drop_time'] = pd.to_datetime(df['drop_time'])

  if test == 0:
    imp_df_train = impute(df)
    df = normalize(imp_df_train)
    
  else:
    df['distance'] = distance(df['pick_lat'], df['drop_lat'], df['pick_lon'], df['drop_lon'])
    df.drop(['pick_lat','drop_lat','pick_lon','drop_lon'], axis=1, inplace=True)
    df.drop(['duration'], axis=1, inplace=True)
    
  return df
  

def __main__():

  #train
  trainDF = pd.read_csv('train.csv', parse_dates=['pickup_time','drop_time'])
  trainDF.label = trainDF.label.map({"correct":1,"incorrect":0})
  
  norm_train = preprocess(trainDF)

  classifier = 1 #KNN->1, Random Forest->2

  classify(norm_train, classifier, 0.8)

  #test
  testDF = pd.read_csv('test.csv', parse_dates=['pickup_time','drop_time'])
  norm_test = preprocess(testDF, test=1)

  preds_rand = predict()
  preds_rand.to_csv('submission_1.csv',index=False)


__main__()


