import sys


def load_data(messages_filepath, categories_filepath):

    '''
    INPUT
    messages_filepath - a String
    categories_filepath - a String


    the function loads the data from the csv file spesified in 
    the args and returns the dataFrame
    
    OUTPUT
    df : DataFrame
    '''

    messages = pd.read_csv('messages_filepath')
    categories = pd.read_csv('categories_filepath')
    df = pd.merge(messages,categories)
    return df


"""__________________________________________________________________________"""

def clean_data(df):

    '''
    INPUT
    df - a DataFrame
    
    the functin first will separte the data to a dataset and labels
    then it will transforme the labels and clean it finnaly it returns the clean data
    
    OUTPUT
    df - DataFrame 
    '''

    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.loc[0]
    category_colnames = list(row.apply(lambda x : x[:-2]))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop(['categories'],axis=1,inplace=True)
    df = pd.concat([df,categories] , axis=1)
    df.drop_duplicates(keep='first',inplace=True)
    return df

"""_____________________________________________________________________________"""


def save_data(df, database_filename):

    '''
    INPUT
    text - a String
    
    the function will make the texte to a loxer case 
    then it will delete all the special caracters 
    after that it will split it to a list of words 
    finally it will lemetize the list of words 
    
    OUTPUT
    a list of tokenized word 
    '''

    engine = create_engine('sqlite:///database_filename.db')
    df.to_sql('Messages', engine, index=False)


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