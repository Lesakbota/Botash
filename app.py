import pandas as pd
import pickle
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Деректерді оқу
movies = pd.read_csv('movies.csv')
movies['combined'] = movies['genres'] + ' ' + movies['description']

# TF-IDF векторизациясы
vectorizer = TfidfVectorizer(stop_words='english')
movie_vectors = vectorizer.fit_transform(movies['combined'])

# Векторизаторды сақтау
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Косинустық ұқсастық
similarity_matrix = cosine_similarity(movie_vectors, movie_vectors)

# Flask қосымшасы
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        title = request.form['title']
        if title in movies['title'].values:
            idx = movies[movies['title'] == title].index[0]
            sims = list(enumerate(similarity_matrix[idx]))
            sorted_sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:6]
            recommendations = [movies.iloc[i[0]]['title'] for i in sorted_sims]
        else:
            recommendations = ['Фильм табылмады.']
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
