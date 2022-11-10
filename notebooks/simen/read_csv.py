import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import h2o
from h2o.automl import H2OAutoML
from shapely.geometry import Point
from simen_funksjoner import *
from read_csv import *
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

#importing stores_train
stores_train = pd.read_csv('/Users/simenvoldqvam/Desktop/Skole/machine_learning/data/raw/stores_train.csv')
stores_train = stores_train[stores_train['year']==2016]


#importing stores_test
stores_test = pd.read_csv('/Users/simenvoldqvam/Desktop/Skole/machine_learning/data/raw/stores_test.csv')
stores_test = stores_test[stores_test['year']==2016]

#importing stores_extra
stores_extra = pd.read_csv('/Users/simenvoldqvam/Desktop/Skole/machine_learning/data/raw/stores_extra.csv')
stores_extra = stores_extra[stores_extra['year']==2016]


#importing grunnkrets_age_distribution
grunnkrets_age_distribution = pd.read_csv("/Users/simenvoldqvam/Desktop/Skole/machine_learning/data/raw/grunnkrets_age_distribution.csv")
grunnkrets_age_distribution = grunnkrets_age_distribution[grunnkrets_age_distribution['year']==2016]
grunnkrets_age_distribution_new = grunnkrets_age_distribution.drop(['year'],axis=1)

#importing grunnkrets_norway_stripped
grunnkrets_norway_stripped = pd.read_csv("/Users/simenvoldqvam/Desktop/Skole/machine_learning/data/raw/grunnkrets_norway_stripped.csv")
grunnkrets_norway_stripped = grunnkrets_norway_stripped[grunnkrets_norway_stripped['year']==2016]
grunnkrets_norway_stripped_new = grunnkrets_norway_stripped.drop(['year'],axis=1)

#importing grunnkrets_households_num_persons
grunnkrets_households_num_persons = pd.read_csv("/Users/simenvoldqvam/Desktop/Skole/machine_learning/data/raw/grunnkrets_households_num_persons.csv")
grunnkrets_households_num_persons = grunnkrets_households_num_persons[grunnkrets_households_num_persons['year']==2016]
grunnkrets_households_num_persons_new = grunnkrets_households_num_persons.drop(['year'],axis=1)

#importing grunnkrets_income_households
grunnkrets_income_households = pd.read_csv("/Users/simenvoldqvam/Desktop/Skole/machine_learning/data/raw/grunnkrets_income_households.csv")
grunnkrets_income_households = grunnkrets_income_households[grunnkrets_income_households['year']==2016]
grunnkrets_income_households_new = grunnkrets_income_households.drop(['year'],axis=1)

#importing plaace_hierarchy
plaace_hierarchy = pd.read_csv("/Users/simenvoldqvam/Desktop/Skole/machine_learning/data/raw/plaace_hierarchy.csv")

#importing test_set_feat
test_set_feat = pd.read_csv("/Users/simenvoldqvam/Desktop/Skole/machine_learning/notebooks/simen/test_set_feat.csv")

#importing train_set_feat
train_set_feat = pd.read_csv("/Users/simenvoldqvam/Desktop/Skole/machine_learning/notebooks/simen/train_set_feat.csv")

#importing stores_gk_impuded
stores_gk_impuded = pd.read_csv("/Users/simenvoldqvam/Desktop/Skole/machine_learning/notebooks/simen/stores_gk_impuded.csv")

#importing stores_test_impuded
stores_test_impuded = pd.read_csv("/Users/simenvoldqvam/Desktop/Skole/machine_learning/notebooks/simen/stores_test_impuded.csv")

