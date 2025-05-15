import pandas as pd
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from transformers import *
import random
from gensim.models.doc2vec import Doc2Vec,\
    TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report

def mlp_classifier():
    assertion_df = pd.read_pickle('target_data/emb_sentiment.pkl')
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                              tags=[str(i)]) for i,
               doc in enumerate(assertion_df['assertion_text'])]
    model = Doc2Vec(vector_size=20,
                min_count=2, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data,
            total_examples=model.corpus_count,
            epochs=model.epochs)
    document_vectors = [model.infer_vector(
    word_tokenize(doc.lower())) for doc in assertion_df['assertion_text']]
    assertion_df['emb_doc'] = document_vectors
    # assertion_df = assertion_df.loc[assertion_df['label']!=0]
    assertion_df['label_derection'] = [0]*len(assertion_df)
    assertion_df.loc[assertion_df['label'] > 0, 'label_derection'] = 1
    assertion_df.loc[assertion_df['label'] < 0, 'label_derection'] = -1
    X_train, X_test, y_train, y_test = train_test_split(assertion_df['embedding'].tolist(), assertion_df['label_derection'], stratify=assertion_df['label_derection'],test_size=0.7,random_state=10)
    clf = MLPClassifier(random_state=42, max_iter=300).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)
    

mlp_classifier()