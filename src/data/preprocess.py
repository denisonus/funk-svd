from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

def preprocess_data(input_file, output_dir, target_ratings=100000, file_suffix="_100K", train_ratio=0.8, random_seed=42,
                    min_user_ratings=20):
    """Preprocess boardgame rating data for recommendation system."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, dtype={"BGGId": "int32", "Rating": "float32", "Username": "string"})

    # Data cleaning
    logger.info(f"Original rows: {len(df)}")
    df = df.drop_duplicates(subset=['Username', 'BGGId'], keep='last')
    df = df.dropna(subset=['Username'])
    df = df[df["Rating"] >= 0.5]

    # User ID mapping
    df["UserId"] = pd.factorize(df["Username"])[0].astype(np.int32)
    
    # Calculate original statistics
    original_users = df["UserId"].nunique()
    original_items = df["BGGId"].nunique()
    original_ratings = len(df)
    original_sparsity = 1 - (original_ratings / (original_users * original_items))
    logger.info(f"Original dataset: {original_users} users, {original_items} items, {original_ratings} ratings")
    logger.info(f"Original sparsity: {original_sparsity:.4f}")
    
    # Filter users with at least min_user_ratings
    user_rating_counts = df["UserId"].value_counts()
    eligible_users = user_rating_counts[user_rating_counts >= min_user_ratings].index
    df = df[df["UserId"].isin(eligible_users)]
    logger.info(f"Users with at least {min_user_ratings} ratings: {len(eligible_users)}")
    
    # Categorize users by rating frequency
    user_rating_counts = df["UserId"].value_counts()
    high_threshold = user_rating_counts.quantile(0.7)
    low_threshold = user_rating_counts.quantile(0.3)
    
    high_raters = user_rating_counts[user_rating_counts >= high_threshold].index
    med_raters = user_rating_counts[(user_rating_counts < high_threshold) & 
                                   (user_rating_counts >= low_threshold)].index
    low_raters = user_rating_counts[(user_rating_counts < low_threshold) & 
                                   (user_rating_counts >= min_user_ratings)].index
    
    logger.info(f"User categories: {len(high_raters)} high, {len(med_raters)} medium, {len(low_raters)} low")
    
    # Set sampling ratios for different user categories
    high_ratio, med_ratio, low_ratio = 0.4, 0.4, 0.2
    
    # Target number of users from each category
    np.random.seed(random_seed)
    total_target_users = min(len(eligible_users), int(target_ratings / user_rating_counts.mean()))
    
    # Sample users from each category
    sampled_high = np.random.choice(high_raters, 
                                   size=min(len(high_raters), int(total_target_users * high_ratio)), 
                                   replace=False)
    sampled_med = np.random.choice(med_raters, 
                                  size=min(len(med_raters), int(total_target_users * med_ratio)), 
                                  replace=False)
    sampled_low = np.random.choice(low_raters, 
                                  size=min(len(low_raters), int(total_target_users * low_ratio)), 
                                  replace=False)
    
    # Combine sampled users
    sampled_users = np.concatenate([sampled_high, sampled_med, sampled_low])
    logger.info(f"Sampled {len(sampled_users)} users: {len(sampled_high)} high, {len(sampled_med)} medium, {len(sampled_low)} low")
    
    # Create reduced dataset with only sampled users
    df_reduced = df[df["UserId"].isin(sampled_users)]
    
    # If we still have too many ratings, sample randomly to reach target
    if len(df_reduced) > target_ratings:
        df_reduced = df_reduced.sample(target_ratings, random_state=random_seed)
    
    logger.info(f"Reduced dataset: {df_reduced['UserId'].nunique()} users, {df_reduced['BGGId'].nunique()} items, {len(df_reduced)} ratings")
    
    # Train-test split ensuring no cold start problems
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    # Split by user to ensure proper distribution
    for user_id, user_data in df_reduced.groupby("UserId"):
        # Ensure each user has at least min_user_ratings in train
        n_ratings = len(user_data)
        n_train = max(min_user_ratings, int(n_ratings * train_ratio))
        
        if n_train >= n_ratings:
            train_df = pd.concat([train_df, user_data])
        else:
            # Shuffle the data for this user
            user_data_shuffled = user_data.sample(frac=1, random_state=random_seed)
            train_df = pd.concat([train_df, user_data_shuffled.iloc[:n_train]])
            test_df = pd.concat([test_df, user_data_shuffled.iloc[n_train:]])
    
    # Ensure no cold-start items in test set
    test_items = set(test_df["BGGId"].unique())
    train_items = set(train_df["BGGId"].unique())
    cold_items = test_items - train_items
    
    if cold_items:
        logger.info(f"Found {len(cold_items)} cold-start items in test set, moving them to train")
        cold_mask = test_df["BGGId"].isin(cold_items)
        # Move cold-start items from test to train
        train_df = pd.concat([train_df, test_df[cold_mask]])
        test_df = test_df[~cold_mask]
    
    # Calculate final metrics
    new_users = df_reduced["UserId"].nunique()
    new_items = df_reduced["BGGId"].nunique()
    new_ratings = len(df_reduced)
    new_sparsity = 1 - (new_ratings / (new_users * new_items))

    logger.info(f"Reduced dataset: {new_sparsity} sparsity")
    logger.info(f"Training set: {len(train_df)} samples ({len(train_df) / len(df_reduced) * 100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} samples ({len(test_df) / len(df_reduced) * 100:.1f}%)")
    logger.info(f"Train users with >= {min_user_ratings} ratings: {(train_df['UserId'].value_counts() >= min_user_ratings).sum()} / {train_df['UserId'].nunique()}")

    # Save datasets
    train_output = output_path / f"user_ratings_train{file_suffix}.csv"
    test_output = output_path / f"user_ratings_test{file_suffix}.csv"
    full_output = output_path / "user_ratings_full.csv"
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    df.to_csv(full_output, index=False)
    logger.info(f"Datasets saved successfully to {output_path}")

    return {"train_file": str(train_output), "test_file": str(test_output), "full_file": str(full_output),
            "metrics": {"users": new_users, "items": new_items, "ratings": new_ratings, "sparsity": new_sparsity}}

if __name__ == "__main__":
    preprocess_data(input_file="../../data/raw/user_ratings.csv", output_dir="../../data/processed",
                    target_ratings=100000, file_suffix="_100K", train_ratio=0.8, random_seed=42, min_user_ratings=20)

