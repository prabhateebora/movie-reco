from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os


def build_similarity_matrix(df):
    """Build the similarity matrix using genres"""
    print("Building similarity matrix...")
    
    
    cv = CountVectorizer(token_pattern=r'[^|]+', lowercase=True)
    
    # Convert all genres into a matrix
    count_matrix = cv.fit_transform(df['genres'])
    
    # Compute cosine similarity between all movies
    sim_matrix = cosine_similarity(count_matrix, count_matrix)
    
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    
    
    return sim_matrix, cv


def save_model(sim_matrix, cv, model_dir='models'):
    """Save the similarity matrix so we can use it later"""
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created {model_dir} directory")
    
    sim_path = os.path.join(model_dir, 'similarity_matrix.pkl')
    cv_path = os.path.join(model_dir, 'count_vectorizer.pkl')
    
    # Save both files using joblib
    joblib.dump(sim_matrix, sim_path)
    print(f"Saved similarity matrix to: {sim_path}")
    
    joblib.dump(cv, cv_path)
    print(f"Saved vectorizer to: {cv_path}")
    
    return sim_path


def load_model(model_dir='models'):
    """Load the saved similarity matrix"""
    sim_path = os.path.join(model_dir, 'similarity_matrix.pkl')
    
    if not os.path.exists(sim_path):
        raise FileNotFoundError(
            f"Model file not found at: {sim_path}\n"
            
        )
    
    print(f"Loading model from: {sim_path}")
    sim_matrix = joblib.load(sim_path)
    print("Model loaded!")
    
    return sim_matrix


def get_recommendations(movie_title, df, sim_matrix, top_n=5):
    """Get movie recommendations based on similarity"""
    
    movie_title = movie_title.strip()
    
    
    matches = df[df['title'].str.lower() == movie_title.lower()]
    
    if matches.empty:
        
        matches = df[df['title'].str.contains(movie_title, case=False, na=False)]
        
        if matches.empty:
            raise ValueError(f"Movie '{movie_title}' not found in the database.")
        
        
        movie_title = matches.iloc[0]['title']
        print(f"Using closest match: {movie_title}")
    
    
    idx = df[df['title'] == movie_title].index[0]
    
    
    scores = list(enumerate(sim_matrix[idx]))

    
    scores = [(i, s) for i, s in scores if i < len(df)]

    sorted_movies = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    
    recommendations = []
    for idx, score in sorted_movies:
        movie_info = {
            'title': df.iloc[idx]['title'],
            'genres': df.iloc[idx]['genres'],
            'similarity_score': round(float(score), 4)
        }
        recommendations.append(movie_info)
    
    return recommendations
