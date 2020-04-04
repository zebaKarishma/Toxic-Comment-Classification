
#Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import re, string
import xgboost
from scipy.sparse import hstack
from sklearn.model_selection import cross_val_score
from sklearn import svm, tree

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=DataConversionWarning, module='sklearn')


#Importing the Dataset
train = pd.read_csv('train.csv',engine='python',error_bad_lines=False).fillna(' ')
test = pd.read_csv('test.csv',engine='python',error_bad_lines=False).fillna(' ')
test_labels = pd.read_csv('test_labels.csv',engine='python',error_bad_lines=False)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

comment_train = train["comment_text"]
comment_test = test["comment_text"]

#Combining both the comments from train and test, to identify tokens.
all_comment = pd.concat([comment_train, comment_test])

#Cleaning the Dataset of empty comments
train['comment_text'].fillna("unknown", inplace=True)
test['comment_text'].fillna("unknown", inplace=True)


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_comment)
train_word_features = word_vectorizer.transform(comment_train)
test_word_features = word_vectorizer.transform(comment_test)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_comment)
train_char_features = char_vectorizer.transform(comment_train)
test_char_features = char_vectorizer.transform(comment_test)

print(train_word_features)

#Combining word tokens and n-grams to form final feature set
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']}) 
for class_name in label_cols:
    train_target = train[[class_name]]
    #classifier = LogisticRegression(C=0.1, solver='sag')
    #classifier = xgboost.XGBClassifier(objective="binary:logistic", random_state=42)
    #classifier = DecisionTreeClassifier(random_state=0,criterion='entropy',max_depth=3)
    classifier = svm.SVC(kernel='rbf')
    print(train_target.shape)
    print(train_features.shape)

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=5))
    scores.append(cv_score)
    print('Logistic Regression - CV score for class {} is {}'.format(class_name, cv_score))


    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]


print('Total CV score is {}'.format(np.mean(scores)))
submission.to_csv('submission2.csv', index=False)