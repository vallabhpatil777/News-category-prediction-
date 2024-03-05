import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os
import seaborn as sns

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

root_dir = "C:/Users/MSI/Downloads/bbc/bbc"

texts = []
categories = []

for category in os.listdir(root_dir):
    category_dir = os.path.join(root_dir, category)
    if os.path.isdir(category_dir):
        for file_name in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.txt'):
                with open(file_path, 'r', encoding='unicode-escape') as file:
                    content = file.read()
                texts.append(content)
                categories.append(category)

df = pd.DataFrame({'text': texts, 'category': categories})

df.to_csv('C:/Users/MSI/Downloads/bbc/bbc/news_articles.csv', index=False)



df = pd.read_csv('C:/Users/MSI/Downloads/bbc/bbc/news_articles.csv')

def preprocess_text(text):
    text = text.lower()  
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

df['processed_text'] = df['text'].apply(preprocess_text)
X= df['processed_text']
y = df["category"]

# Feature Extraction
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()
ngram_vectorizer = CountVectorizer(ngram_range=(2,2))

X_count = count_vectorizer.fit_transform(X)
X_tfidf = tfidf_vectorizer.fit_transform(X)
X_ngrams = ngram_vectorizer.fit_transform(X)

X_combined = pd.concat([pd.DataFrame(X_tfidf.toarray()),pd.DataFrame(X_count.toarray()),pd.DataFrame(X_ngrams.toarray())],axis=1)  

selector = SelectKBest(chi2,k=500)
X_selected = selector.fit_transform(X_combined,y)

X_train, X_val, y_train, y_val = train_test_split(X_selected,y, test_size=0.3, random_state=23)
X_val , X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=23) 

model = SVC()


param_grid = {'C' : [0.1,0.5,1,10] , 'gamma' : [0.01,0.1,1,1.5], 'kernel' : ['linear','rbf', 'poly']}

cv = KFold(n_splits=5, shuffle=True, random_state=23)
grid_search = GridSearchCV(estimator = model, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=3)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
val_score = best_model.score(X_val,y_val)

print("Best parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n Accuracy :", accuracy)

print("\n Classification Report :")
print(classification_report(y_test, y_pred))

confusion_mat = confusion_matrix(y_test,y_pred)

sns.heatmap(confusion_mat,annot=True, cmap='Blues',xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.show()