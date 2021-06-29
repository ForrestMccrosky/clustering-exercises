import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split

from env import host, user, password



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
    
    return df

############################## Train/Validate/Split Function #########################


def train_validate_test_split(df):
    '''split df into train, validate, test'''
    
    train, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    return train, validate, test



################################# SQL connect Function ##############################


def sql_connect(db, user=user, host=host, password=password):
    '''
    This function allows me to connect the Codeup database to pull SQL tables
    Using private information from my env.py file.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    

####################### Acquire Zillow Database Function ##########################


def acquire_zillow():
    ''' 
    This function acquires the zillow database from SQL into Pandas and filters the data
    according to the project scope of 2017 purchased properties and the most recent transactions
    to avoid duplicates
    '''
    
    sql_query = '''
    select * 
    from predictions_2017
    left join properties_2017 using(parcelid)
    left join airconditioningtype using(airconditioningtypeid)
    left join architecturalstyletype using(architecturalstyletypeid)
    left join buildingclasstype using(buildingclasstypeid)
    left join heatingorsystemtype using(heatingorsystemtypeid)
    left join propertylandusetype using(propertylandusetypeid)
    left join storytype using(storytypeid)
    left join typeconstructiontype using(typeconstructiontypeid)
    where latitude is not null and longitude is not null
    '''
    
    df = pd.read_sql(sql_query, sql_connect('zillow'))
    
    
    ## filtering for properties that single residential properties
    df = df[df['propertylandusetypeid'] == 261]
    
    ##getting rid of duplicate columns
    df= df.loc[:, ~df.columns.duplicated()]
    
    ## drop duplicate parcelids keeping the latest transaction from 2017
    df = df.sort_values('transactiondate').drop_duplicates('parcelid',keep='last') 
    
    return df 


###################### Function To Get County Names ###########################


def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    ## Making county column based on fips column using the associated ID's

    df['county'] = df['fips'].map({6037: 'Los Angeles', 
                                             6059: 'Orange', 
                                             6111: 'Ventura'})
    return df

###################### Function To Create More features ###########################


def create_features(df):
    '''
    Compute new features out of existing features in order to reduce noise, capture signals, 
    and reduce collinearity, or dependence between independent variables.
    
    features computed:
    age
    age_bin
    taxrate
    acres
    acres_bin
    sqft_bin
    bed_bath_ratio
    '''
    df['age'] = 2017 - df.yearbuilt
    df['age_bin'] = pd.cut(df.age, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                                   130,140],
                           labels = [0, .066, .133, .20, .266, .333, .40, .466, .533, 
                                     .60, .666, .733, .8, .866, .933])

    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560

    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200],
                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    # square feet bin
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet, 
                            bins = [0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000,
                                    12000],
                            labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])


    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',})


    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    return df

