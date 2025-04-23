import numpy as np
from typing import List, Dict, Set, Any
from src.config.settings import EVALUATION_CONFIG


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
            dcg += rel / np.log2(i + 2)
    
    ideal_items = sorted(relevant_items.items(), key=lambda x: x[1], reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, (_, rel) in enumerate(ideal_items[:k]))
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_coverage(recommended_items_per_user: List[List[int]], catalog_size: int) -> float:
    """Calculate catalog coverage"""
    if not recommended_items_per_user or catalog_size <= 0:
        return 0.0
    
    all_recommended = set()
    for user_recommendations in recommended_items_per_user:
        all_recommended.update(user_recommendations)
    
    return len(all_recommended) / catalog_size


def evaluate_recommendations(model: Any, test_data: Any) -> Dict[str, Dict[int, float]]:
    """Evaluate a recommendation models using multiple metrics"""
    k_values = EVALUATION_CONFIG['k_values']
    relevance_threshold = EVALUATION_CONFIG['relevance_threshold']
    
    user_item_relevance = {}
    all_items = set()
    
    for _, row in test_data.iterrows():
        user_id, item_id, rating = row['UserId'], row['BGGId'], row['Rating']
            
        if user_id not in user_item_relevance:
            user_item_relevance[user_id] = {}
        
        user_item_relevance[user_id][item_id] = rating
        all_items.add(item_id)
    
    result_metrics = {
        'precision': {k: 0.0 for k in k_values},
        'recall': {k: 0.0 for k in k_values},
        'ndcg': {k: 0.0 for k in k_values},
    }
    
    user_recommendations = []
    users_with_relevant_items = 0
    
    for user_id, item_ratings in user_item_relevance.items():
        relevant_items_dict = {item_id: rating for item_id, rating in item_ratings.items() 
                              if rating >= relevance_threshold}
        
        if not relevant_items_dict:
            continue
            
        users_with_relevant_items += 1
        relevant_items_set = set(relevant_items_dict.keys())
        
        candidate_items = model.item_ids.copy()
        
        predictions = model.predict_for_user(user_id, candidate_items)
        
        recommended_items = [item_id for item_id, _ in 
                            sorted(predictions.items(), key=lambda x: x[1], reverse=True)]
        
        user_recommendations.append(recommended_items[:max(k_values)])
        
        for k in k_values:
            result_metrics['precision'][k] += precision_at_k(recommended_items, relevant_items_set, k)
            result_metrics['recall'][k] += recall_at_k(recommended_items, relevant_items_set, k)
            result_metrics['ndcg'][k] += ndcg_at_k(recommended_items, relevant_items_dict, k)
    
    coverage = calculate_coverage(user_recommendations, len(all_items))
    result_metrics['coverage'] = {k: coverage for k in k_values}
    
    if users_with_relevant_items > 0:
        for metric in ['precision', 'recall', 'ndcg']:
            for k in k_values:
                result_metrics[metric][k] /= users_with_relevant_items
    
    return result_metrics