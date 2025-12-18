from flask import Flask, request, jsonify
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import load_and_clean_data
from recommendation import load_model, get_recommendations

app = Flask(__name__)

# Store the data and model in memory
movies_df = None
sim_matrix = None


def initialize_app():
    """Load the data and model when the app starts"""
    global movies_df, sim_matrix
    
    print("\n" + "="*60)
    print("Starting Movie Recommendation API")
    print("="*60 + "\n")
    
    try:
        data_path = os.path.join('data', 'movies.csv')
        print(f"Loading movies from: {data_path}")
        movies_df = load_and_clean_data(data_path)
        print(f"✓ Loaded {len(movies_df)} movies\n")
        
        print("Loading trained model...")
        sim_matrix = load_model(model_dir='models')
        print("✓ Model ready\n")
        
        print("="*60)
        print("API is ready!")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}\n")
        print("Make sure:")
        print("  1. movies.csv is in data/ folder")
        print("  2. You ran 'python train_model.py' first")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@app.route('/')
def home():
    """Home page - shows API info"""
    return jsonify({
        "message": "Movie Recommendation API",
        "version": "1.0",
        "endpoints": {
            "/recommend": {
                "method": "POST",
                "description": "Get movie recommendations",
                "input": {"movie_title": "string"},
                "example": {"movie_title": "Toy Story (1995)"}
            }
        },
        "status": "running",
        "total_movies": len(movies_df) if movies_df is not None else 0
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    """Main endpoint for getting recommendations"""
    try:
        # Make sure everything is loaded
        if movies_df is None or sim_matrix is None:
            return jsonify({
                "error": "Service not ready",
                "message": "Please restart the server"
            }), 500
        
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({
                "error": "Invalid request",
                "message": "Please send JSON data"
            }), 400
        
        data = request.get_json()
        
        # Make sure movie_title is in the request
        if 'movie_title' not in data:
            return jsonify({
                "error": "Missing field",
                "message": "Need 'movie_title' in your request",
                "example": {"movie_title": "Toy Story (1995)"}
            }), 400
        
        movie_title = data['movie_title']
        
        # Check if movie_title is empty
        if not movie_title or not movie_title.strip():
            return jsonify({
                "error": "Invalid input",
                "message": "movie_title can't be empty"
            }), 400
        
        # Get the recommendations
        recommendations = get_recommendations(
            movie_title=movie_title,
            df=movies_df,
            sim_matrix=sim_matrix,
            top_n=5
        )
        
        # Send back the results
        response = {
            "query": movie_title,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        # Movie not found
        return jsonify({
            "error": "Movie not found",
            "message": str(e),
            "suggestion": "Try including the year in the title"
        }), 404
        
    except Exception as e:
        # Something unexpected happened
        print(f"Error in /recommend: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": "Server error",
            "message": "Something went wrong"
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "message": "This endpoint doesn't exist",
        "available_endpoints": ["/", "/recommend"]
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error": "Method not allowed",
        "message": "Use POST for /recommend"
    }), 405


if __name__ == "__main__":
    # Load everything first
    initialize_app()
    
    # Start the server
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
