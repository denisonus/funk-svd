from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def preprocess_data(input_file, output_dir, target_ratings=100000, file_suffix="_100K", train_ratio=0.8,
                    random_seed=42):
    """
    Preprocess boardgame rating data for recommendation system.

    Args:
        input_file: Path to the raw data CSV
        output_dir: Directory to save processed files
        target_ratings: Target number of ratings in the reduced dataset
        file_suffix: Suffix to add to output filenames
        train_ratio: Ratio of data to use for training
        random_seed: Random seed for reproducibility
    """

    # Make sure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, dtype={"BGGId": "int32", "Rating": "float32", "Username": "string"})

    # Data cleaning
    logger.info(f"Original rows: {len(df)}")
    logger.info(f"Duplicated rows: {df.duplicated().sum()}")
    logger.info(f"Duplicated user ratings: {df.duplicated(subset=['Username', 'BGGId'], keep=False).sum()}")
    df = df.drop_duplicates(subset=['Username', 'BGGId'], keep='last')

    logger.info(f"Missing values:\n{df.isnull().sum()}")
    df = df.dropna(subset=['Username'])

    logger.info(f"Rating statistics:\n{df['Rating'].describe()}")
    logger.info(f"Ratings < 0.5: {(df['Rating'] < 0.5).sum()}")
    df = df[df["Rating"] >= 0.5]

    # User ID mapping
    df["UserId"] = pd.factorize(df["Username"])[0].astype(np.int32)

    # Calculate original metrics
    original_users = df['UserId'].nunique()
    original_items = df['BGGId'].nunique()
    original_ratings = len(df)
    original_matrix_size = original_users * original_items
    original_sparsity = 1 - (original_ratings / original_matrix_size)

    logger.info("Original metrics:")
    logger.info(f"Users: {original_users} | Items: {original_items} | Ratings: {original_ratings}")
    logger.info(f"Matrix sparsity: {original_sparsity:.4f} ({original_sparsity * 100:.2f}%)")

    # Calculate target metrics
    target_matrix_size = target_ratings / (1 - original_sparsity)
    scaling_factor = (target_matrix_size / original_matrix_size) ** 0.5
    target_users = int(original_users * scaling_factor)
    target_items = int(original_items * scaling_factor)

    logger.info(f"Scaling factor: {scaling_factor}")
    logger.info(f"Target users: {target_users} | Target items: {target_items}")

    # Filter most active users and most rated items
    most_active_users = set(df['UserId'].value_counts().nlargest(target_users).index)
    most_rated_items = set(df['BGGId'].value_counts().nlargest(target_items).index)
    df_reduced = df[df['UserId'].isin(most_active_users) & df['BGGId'].isin(most_rated_items)]

    # Sample if we have more ratings than needed
    if len(df_reduced) > target_ratings:
        df_reduced = df_reduced.sample(n=target_ratings, random_state=random_seed)

    # Calculate final metrics
    new_users = df_reduced['UserId'].nunique()
    new_items = df_reduced['BGGId'].nunique()
    new_ratings = len(df_reduced)
    new_sparsity = 1 - (new_ratings / (new_users * new_items))

    logger.info("Final metrics:")
    logger.info(f"Users: {new_users} | Items: {new_items} | Ratings: {new_ratings}")
    logger.info(f"Matrix sparsity: {new_sparsity:.4f} ({new_sparsity * 100:.2f}%)")

    # Split data into train and test sets
    np.random.seed(random_seed)

    # Group by UserId
    user_groups = df_reduced.groupby('UserId')
    train_indices = []
    test_indices = []

    # For each user, put some ratings in train and some in test
    for user_id, user_df in user_groups:
        indices = user_df.index.tolist()

        # If user has only one rating, put it in training
        if len(indices) == 1:
            train_indices.extend(indices)
            continue

        # Shuffle the user's ratings
        np.random.shuffle(indices)

        # Split point for this user
        user_split = max(1, int(train_ratio * len(indices)))

        # Add to train and test sets
        train_indices.extend(indices[:user_split])
        test_indices.extend(indices[user_split:])

    train_df = df_reduced.loc[train_indices]
    test_df = df_reduced.loc[test_indices]

    logger.info(f"Training set: {len(train_df)} samples ({len(train_df) / len(df_reduced) * 100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} samples ({len(test_df) / len(df_reduced) * 100:.1f}%)")

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
                    target_ratings=100000,
                    file_suffix="_100K", train_ratio=0.8, random_seed=42)
