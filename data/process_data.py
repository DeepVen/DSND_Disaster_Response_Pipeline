import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    '''
    1) Load data from (2) source files in the project folder using Pandas Dataframe
    2) Merge data from 2 dataframes into 1 based on the common 'id' field    
    Args:
    messages_file_path: Messages CSV file
    categories_file_path: Categories CSV file    
    Returns:
    Merged dataframe containing data from both input files    
    '''
    
    # load data into python memory using Pandas 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='inner', on='id')
    
    return df


def clean_data(df):
    '''
    Clean the Dataframe as part of preprocessing to get it ready ML pipeline    
    Args:
    Dataframe from load_data method 
    Returns:
    Dataframe containing Cleaned data
    '''    
    categories = df.categories.str.split(';', expand=True)
    categories_column_header = categories.loc[0].values
    categories_column_header = [x[:-2] for x in categories_column_header]
    categories.columns = categories_column_header
    
    # do element wise changes on dataframe using applymap
    categories = categories.applymap(lambda x : int(x[-1]))
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # check number of duplicates
    print ('# of duplicates in original file: ', df[df.duplicated(subset='message')]['message'].count())

    # drop duplicates
    df.drop_duplicates(subset='message', inplace=True)
    
    # check number of duplicates
    print ('# of duplicates after cleaning(/dropping): ', df[df.duplicated(subset='message')]['message'].count())
    
    return df

def save_data(df, database_filename):
    '''
    Save the cleaned data from Dataframe into a SQLite file for further processing by ML pipeline
    Args:
    Dataframe from clean_data method 
    Returns:
    N.A.
    '''    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('message_source', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()