import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import load_and_clean_data, get_data_summary
from recommendation import build_similarity_matrix, save_model


def main():
    print("\n" + "="*60)
    print("MOVIE RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("="*60 + "\n")
    
    data_path = os.path.join('data', 'movies.csv')
    
    try:
        # Load and clean the movie data
        print("Step 1: Loading and cleaning data...")
        df = load_and_clean_data(data_path)
        
        get_data_summary(df)
        
        # Build the similarity matrix
        print("\nStep 2: Building recommendation model...")
        sim_matrix, cv = build_similarity_matrix(df)
        
        # Save everything
        print("\nStep 3: Saving model...")
        save_model(sim_matrix, cv, model_dir='models')
        
        print("\n" + "="*60)
        print("SUCCESS! Model training complete")
        print("="*60)
        print(f"✓ Processed {len(df)} movies")
        print(f"✓ Model saved to models/")
        print(f"✓ You can now run: python app.py")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure 'movies.csv' is in the 'data/' folder")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Your CSV needs these columns: movieId, title, genres")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
