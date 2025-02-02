import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Read 2 dataset and combine them after pre processing
    INPUT:
        messages_filepath: path of message dataset
        categories_filepath: path of categories dataset
    OUTPUT:
        df: dataframe combine from message and categories after preprocessing
    """
    # load dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge dataset
    df = messages.merge(categories, on = 'id', how = 'inner')
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories[0:1]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.str[:-2]).values[0].tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])   
    # replace 2 by 1
    categories['related'] = categories['related'].replace(to_replace=2, value=1)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    return df    

def clean_data(df):
    """
    Drop duplicates data
    INPUT: 
        df: dataframe before drop duplicate
    OUTPUT: 
        df: dataframe after drop duplicate
    """
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Save data to SQLite
    INPUT: 
        df: dataframe use to convert into SQLite
        database_filename: name of database
    OUTPUT:
        Return Database like database_filename
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False)  


def main():
    """
    Combine all step above to Load, Clean and Save data
    """
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