import pandas as pd
import os

def load_and_clean_data(file_path):
    """Load and clean the movie dataset"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Making sure we have the columns we need
    required_cols = ['movieId', 'title', 'genres']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    
    df = df[required_cols].copy()
    
    # Clean up missing values
    if df['movieId'].isnull().any():
        print("Warning: Some movies are missing IDs, filling them...")
        df['movieId'].fillna(method='ffill', inplace=True)
    
    if df['title'].isnull().any():
        print("Warning: Some movies have no title, filling with 'Unknown'")
        df['title'].fillna('Unknown', inplace=True)
    
    if df['genres'].isnull().any():
        print("Warning: Some movies have no genres listed")
        df['genres'].fillna('(no genres listed)', inplace=True)
    
    # Remove duplicates - keep the first occurrence
    before = len(df)
    df = df.drop_duplicates(subset=['title'], keep='first')
    after = len(df)
    if before > after:
        print(f"Removed {before - after} duplicate movies")
    
    df = df.reset_index(drop=True)
    print(f"Cleaned data successfully. Total movies: {len(df)}")
    
    return df


def get_data_summary(df):
    """Print some basic info about the dataset"""
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total movies: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head(3))
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print("="*50 + "\n")
