import re
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.data.path.append('C:\\Users\\LENOVO\\nltk_data')
import streamlit as st
import pandas as pd
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, SnowballStemmer
from pathlib import Path

try:
    nltk.data.find('tokenizers/punkt')
    print("Punkt est accessible.")
except LookupError:
    print("Punkt n'est pas accessible.")

    nltk.download('punkt')

script_dir = Path(__file__).parent

file_path = script_dir / 'file.json'

if file_path.exists():
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        df = pd.DataFrame(data)
else:
    print(f"Le fichier {file_path} n'a pas été trouvé.")
    exit(1)

df['tokens'] = df['question'].apply(word_tokenize)

stop_words_english = set(stopwords.words('english'))
stop_words_french = set(stopwords.words('french'))
stop_words = stop_words_english.union(stop_words_french)

df['filtered_tokens'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

df['pos_tags'] = df['tokens'].apply(lambda tokens: pos_tag(tokens))

stemmer_french = SnowballStemmer('french')
stemmer_english = SnowballStemmer('english')

df['stemmed_french'] = df['filtered_tokens'].apply(lambda tokens:[stemmer_french.stem(word) for word in tokens])
df['stemmed_english'] = df['filtered_tokens'].apply(lambda tokens:[stemmer_english.stem(word) for word in tokens])

legitimatise = WordNetLemmatizer()
df['legitimatise_tokens_french'] = df['stemmed_french'].apply(lambda tokens: [legitimatise.lemmatize(word) for word in tokens])
df['legitimatise_tokens_english'] = df['stemmed_english'].apply(lambda tokens: [legitimatise.lemmatize(word) for word in tokens])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

corpus = df['question']
labels = df['category']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

from sklearn.model_selection import GridSearchCV

para = {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 15]}
grid_dt = GridSearchCV(DecisionTreeClassifier(), para, cv=3)
grid_dt.fit(X_train, y_train)

para_nb = {'alpha': [0.1, 0.5, 1.0]}
grid_nb = GridSearchCV(MultinomialNB(), para_nb, cv=3)
grid_nb.fit(X_train, y_train)


from fuzzywuzzy import fuzz
from nltk.chat.util import Chat, reflections

def charger_pair_json():
    pair = []
    with open(file_path, 'r', encoding='utf-8') as fl:
        json_data = json.load(fl)

    for item in json_data:
        if isinstance(item, dict):
            question = item.get("question")
            cate = item.get("category")

            if question and cate:
                pair.append((question.lower(), cate))

    return pair

file = "C:\\Users\\LENOVO\\Project_final\\file.json"
pairs = charger_pair_json()

chatbot =Chat(pairs,reflections)

def get_category(question_user):
    question_user = question_user.lower()
    best_match = None
    highest_score = 0

    for question, question_category in pairs:
        score = fuzz.ratio(question_user, question)
        if score > highest_score:
            highest_score = score
            best_match = question_category

    if highest_score >= 70:
        return best_match
    return None

st.title("Chatbot de Classification Service")

user_input = st.text_input("Ask your question here :")

if user_input:

    user_input = re.sub(r'[^\w\s]', '', user_input.lower())

    category = get_category(user_input)

    if category:
            st.write(f"le service concerne est: {category} ")
    else :
            st.write("Chatbot: Sorry, I don't understand your question..")


