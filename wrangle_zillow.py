import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split



############################## Function File ###############################

############################## Summarize DF function #######################


def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('=====================================================')
    print('Dataframe head: ')
    print(df.head(3))
    print('=====================================================')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================')
    print('Dataframe Description: ')
    print(df.describe())
    

########################## Value_Counts DF function #######################

def df_value_counts(df):

    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    
    print('DataFrame value counts: ')
    print('------------------------------------------')
    print('')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
            print('-------------------------------------------')
            print('')

############################## Histogram DF function #########################

def histogram_df(df):

    for col in df.columns:
        df[col].hist()
        plt.ylabel('Frequency')
        plt.xlabel(f'{col} value')
        plt.title(f'Distribution of {col}')
        plt.show()
        
        
############################## Missing Values DF functions #########################


def nulls_by_col(df):
    '''
    This function takes in a dataframe and provides a percentage look 
    at the columns with null values (values, and percent missing)
    '''
    
    num_missing = df.isnull().sum()
    pct_missing = df.isnull().sum()/df.shape[0]
    
    df = pd.DataFrame({'num_rows_missing': num_missing, 
                       'pct_rows_missing': pct_missing}) # create dataframe using variables
    return df


def nulls_by_row(df):
    '''take in a dataframe 
       get count of missing columns per row
       percent of missing columns per row 
       and number of rows missing the same number of columns
       in a dataframe'''
    
    num_cols_missing = df.isnull().sum(axis=1) # number of columns that are missing in each row
    
    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100  # percent of columns missing in each row 
    
    # create a dataframe for the series and reset the index creating an index column
    # group by count of both columns, turns index column into a count of matching rows
    # change the index name and reset the index
    
    return (pd.DataFrame({'num_cols_missing': num_cols_missing, 'pct_cols_missing': pct_cols_missing}).reset_index()
            .groupby(['num_cols_missing','pct_cols_missing']).count()
            .rename(index=str, columns={'index': 'num_rows'}).reset_index())


############################## Remove Outliers Function #########################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[f'{col}'].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]
        
    return df

############################## Handle Missing Values Function #########################


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .5):
    ''' 
        take in a dataframe and a proportion for columns and rows
        return dataframe with columns and rows not meeting proportions dropped
    '''
    col_thresh = int(round(prop_required_column*df.shape[0],0)) # calc column threshold
    
    df.dropna(axis=1, thresh=col_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    row_thresh = int(round(prop_required_row*df.shape[1],0))  # calc row threshhold
    
    df.dropna(axis=0, thresh=row_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    return df

############################## Impute Columns Function #########################


# impute columns *do this after you split*

def impute(df, my_strategy, column_list):
    ''' take in a df, strategy, and cloumn list
        return df with listed columns imputed using input stratagy
    '''
        
    imputer = SimpleImputer(strategy=my_strategy)  # build imputer

    df[column_list] = imputer.fit_transform(df[column_list]) # fit/transform selected columns

    return df

############################## Prepare Zillow Function #########################



def prepare_zillow(df):
    ''' Prepare Zillow Data'''
    
    # Restrict propertylandusedesc to those of single unit
    df = df[(df.propertylandusedesc == 261) |
          (df.propertylandusedesc == 263) |
          (df.propertylandusedesc == 275) |
          (df.propertylandusedesc == 264)]
    
    # remove outliers in bed count, bath count, and area to better target single unit properties
    df = remove_outliers(df, 1.5, ['calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt'])
    
    # dropping cols/rows where more than half of the values are null
    df = handle_missing_values(df, prop_required_column = .5, prop_required_row = .5)
    
    # dropping the columns with 17K missing values too much to fill/impute/drop rows
    df = df.drop(columns=['heatingorsystemtypeid', 'buildingqualitytypeid', 'unitcnt'])
    
    # imputing descreet columns with most frequent value
    df = impute(df, 'most_frequent', ['calculatedbathnbr', 'fullbathcnt', 'regionidcity', 'regionidzip', 'yearbuilt', 'censustractandblock'])
    
    # imputing continuous columns with median value
    df = impute(df, 'median', ['finishedsquarefeet12', 'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount'])
    
    return df

############################## Train/Validate/Split Function #########################


def train_validate_test_split(df):
    '''split df into train, validate, test'''
    
    train, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    return train, validate, test
