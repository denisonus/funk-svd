from typing import List, Dict, Set

import numpy as np
import pandas as pd

from src.config.settings import EVALUATION_CONFIG
from src.models.funk_svd import FunkSVD


def precision_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """Calculate Precision@k"""
    if k <= 0 or not recommended_items or not relevant_items:
        return 0.0
        
    top_k = recommended_items[:k]
    hits = sum(1 for item in top_k if item in relevant_items)
    return hits / min(k, len(recommended_items))


def recall_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """Calculate Recall@k"""
    if k <= 0 or not recommended_items or not relevant_items:
        return 0.0
        
    top_k = recommended_items[:k]
    hits = sum(1 for item in top_k if item in relevant_items)
    return hits / len(relevant_items) if relevant_items else 0.0


def ndcg_at_k(recommended_items: List[int], relevant_items: Dict[int, float], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k (NDCG@k)"""
    if k <= 0 or not recommended_items or not relevant_items:
        return 0.0

    dcg = 0.0
    for i, item_id in enumerate(recommended_items[:k]):
        if item_id in relevant_items:
            rel = relevant_items[item_id]
            gain = 2**rel - 1
            dcg += gain / np.log2(i + 2)

    ideal_items = sorted(relevant_items.items(), key=lambda x: x[1], reverse=True)
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, (_, rel) in enumerate(ideal_items[:k]))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(model: FunkSVD, test_data: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    """Evaluate a recommendation model using multiple metrics, focusing only on rated test items"""
    k_values = EVALUATION_CONFIG['k_values']
    relevance_threshold = EVALUATION_CONFIG['relevance_threshold']
    
    metrics = {
        'precision': {k: 0.0 for k in k_values},
        'recall': {k: 0.0 for k in k_values},
        'ndcg': {k: 0.0 for k in k_values}
    }
    
    users_evaluated = 0
    
    for user_id in test_data['UserId'].unique():
        if user_id not in model.user_id_to_idx:
            continue

        user_test_data = test_data[test_data['UserId'] == user_id]

        item_ratings = {row['BGGId']: row['Rating'] for _, row in user_test_data.iterrows()}
        relevant_items_dict = {item_id: rating for item_id, rating in item_ratings.items()
                              if rating >= relevance_threshold}
        
        if not relevant_items_dict:
            continue
            
        users_evaluated += 1
        relevant_items_set = set(relevant_items_dict.keys())
        
        predictions = model.predict_for_user(user_id, list(item_ratings.keys()))
        recommended_items = [item_id for item_id, _ in sorted(predictions.items(), key=lambda x: x[1], reverse=True)]
        
        for k in k_values:
            metrics['precision'][k] += precision_at_k(recommended_items, relevant_items_set, k)
            metrics['recall'][k] += recall_at_k(recommended_items, relevant_items_set, k)
            metrics['ndcg'][k] += ndcg_at_k(recommended_items, relevant_items_dict, k)
    
    if users_evaluated > 0:
        for metric in metrics:
            for k in k_values:
                metrics[metric][k] /= users_evaluated

    return metrics
