import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

nltk.download('stopwords')

def load_data(database_filepath):
    '''
    Load data from SQLite file into a Dataframe and store data into Feature and Target Dataframes   
    Args:
    database_filepath: SQLite file path 
    Returns:
    Feature Dataframe
    Target Dataframe
    Target labels
    '''    
    
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('message_source', engine)
    
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    
    # introducing new condition to handle 1 irrelevant data/value w.r.t 'related' field i.e. replace 2 by 1    
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    
    category_names = Y.columns
    
    return X, Y, category_names 

def tokenize(text):
    '''
    Tokenize text data for further processing
    Args:
    text: Messages 
    Returns:
    clean_tokens: Tokenizing and lemmatizing of text and stop words removed.
    '''
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    stop_words = nltk.corpus.stopwords.words("english")

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_token = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    '''
    Define ML model
    Args:
    N.A. 
    Returns:
    cv: Defined model
    '''
    pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    parameters = {    
        'tfidf__max_features': (None, 10000),    
        'tfidf__max_df': (0.8, 1.0) 
        }
    
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model and print performance metrics
    Args:
    model: Trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    Returns:
    N.A.
    '''
    # predict
    pred = model.predict(X_test)

    for i in range(36):
        print(Y_test.columns[i], " - Accuracy score: ", accuracy_score(Y_test.values[:,i],pred[:,i]) , ' - Precision score: ' , precision_score(Y_test.values[:,i],pred[:,i]) , " - Recall score: " ,  recall_score(Y_test.values[:,i],pred[:,i]) , " - F1 score: " ,   f1_score(Y_test.values[:,i],pred[:,i]), ' \n')
        
  

def save_model(model, model_filepath):
    """
    Saves trained and evaluated model to a pickle file    
    Args:
    model: Trained model
    model_filepath: Filepath to save the model
    Returns:
    N.A.
    """
    
    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()