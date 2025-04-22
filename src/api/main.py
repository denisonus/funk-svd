from asyncio import Lock
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends

from schemas import *
from src.config import FUNK_SVD_MODEL_DIR
from src.data.load_dataset import get_train_data, get_games_data
from src.recommender import GameRecommender

# Global recommender instance
instance = None
# Create a global lock for model updates
model_lock = Lock()

# Define lifespan context manager
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: initialize the recommender
    global instance
    if instance is None:
        train_data = get_train_data()
        games_data = get_games_data()
        instance = GameRecommender(train_data, games_data)
        instance.load(FUNK_SVD_MODEL_DIR)

    
    yield  # This yields control back to FastAPI
    
    # Shutdown: save model state if needed
    if instance is not None:
        instance.save(FUNK_SVD_MODEL_DIR)

# Initialize app with lifespan context manager
app = FastAPI(
    title="Game Recommender API",
    description="API for board game recommendations using FunkSVD algorithm",
    version="1.0.0",
    lifespan=lifespan
)

# Dependency to get the recommender instance
async def get_recommender():
    global instance
    if instance is None:
        raise HTTPException(
            status_code=500,
            detail="Recommender model not initialized. The service may still be starting up or encountered an initialization error."
        )
            
    return instance

# API routes
@app.get("/", response_model=StatusResponse)
async def root(recommender: GameRecommender = Depends(get_recommender)):
    """Get API status and model information"""
    return {
        "status": "ok",
        "message": "Game recommender API is running",
        "model_info": {
            "n_users": len(recommender.model.user_ids),
            "n_items": len(recommender.model.item_ids),
            "n_factors": recommender.model.n_factors
        }
    }

@app.post("/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(
    request: RecommendationRequest,
    recommender: GameRecommender = Depends(get_recommender)
):
    """Get personalized recommendations for a user"""
    attributes = ["Name", "YearPublished", "Description", "ImageURL"] if request.include_details else None
    
    try:
        recommendations = recommender.get_recommendations(
            user_id=request.user_id,
            n=request.num_recommendations,
            attributes=attributes
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error getting recommendations: {str(e)}")

@app.get("/games", response_model=List[GameDetails])
async def get_games(
    request: GamesListRequest = Depends(),
    recommender: GameRecommender = Depends(get_recommender)
):
    """Get a paginated list of games with optional filtering and sorting"""
    games = recommender.get_all_games()
    
    # Apply name filter if provided
    if request.name_filter:
        games = [g for g in games if request.name_filter.lower() in g.get('Name', '').lower()]
    
    # Apply sorting using the helper function
    games.sort(key=lambda x: get_sort_key(x, request.sort_by.value), reverse=request.sort_desc)
    
    # Apply pagination
    start_idx = (request.page - 1) * request.page_size
    end_idx = start_idx + request.page_size
    
    return games[start_idx:end_idx]

@app.get("/games/{game_id}", response_model=GameDetails)
async def get_game_details(
    game_id: int,
    recommender: GameRecommender = Depends(get_recommender)
):
    """Get details for a specific game"""
    game_details = recommender.get_game_details(game_id)
    if not game_details:
        raise HTTPException(status_code=404, detail=f"Game with ID {game_id} not found")
    return game_details

# Helper function for sorting games
def get_sort_key(game_dict: Dict[str, Any], sort_field: str) -> Any:
    """Helper function to get the sorting key for a game dictionary."""
    if sort_field == SortOption.POPULARITY.value:
        # Calculate popularity score (example: rating * num_ratings)
        # Ensure 'NumRatings' exists or provide a default
        return game_dict.get('AverageRating', 0.0) * game_dict.get('NumRatings', 0) 
    elif sort_field == SortOption.YEAR.value:
        return game_dict.get(sort_field, 0)  # Default year to 0
    elif sort_field == SortOption.RATING.value:
        return game_dict.get('AverageRating', 0.0) # Use AverageRating for sorting by rating
    elif sort_field == SortOption.NAME.value:
        return game_dict.get(sort_field, '') # Default name to empty string
    else:
        # Fallback for any other unexpected sort field
        return game_dict.get(sort_field, '')

@app.post("/ratings", response_model=RatingUpdateResponse)
async def add_ratings(
    ratings_data: UserRatings, 
    background_tasks: BackgroundTasks,
    recommender: GameRecommender = Depends(get_recommender)
):
    """Add new ratings from a user and update the model"""
    # Update to use model_dump() instead of dict() for Pydantic v2 compatibility
    ratings_list = [rating.model_dump() for rating in ratings_data.ratings]
    
    # Immediate validation and user ID assignment
    async with model_lock:
        success, user_id, is_new_user = recommender.add_ratings(
            ratings=ratings_list,
            user_id=ratings_data.UserId
        )
    
    # Save the updated model in the background
    background_tasks.add_task(save_model_in_background, recommender)
    
    return {
        "success": success,
        "user_id": user_id,
        "is_new_user": is_new_user,
        "num_ratings_processed": len(ratings_list)
    }

@app.get("/popular", response_model=List[RecommendationResponse])
async def get_popular_recommendations(
    request: PopularRecommendationRequest = Depends(),
    recommender: GameRecommender = Depends(get_recommender)
):
    """Get popular game recommendations based on overall ratings"""
    popular_items = recommender.get_popular_recommendations(
        recommender.train_data, 
        n=request.num_recommendations
    )
    
    result = []
    attributes_to_include = ["Name", "YearPublished", "Description", "ImageURL"]
    for item_id, popularity in popular_items:
        rec = {"BGGId": item_id, "PredictedRating": popularity}
        
        # Add game details if requested and available
        if request.include_details and recommender.games_data and item_id in recommender.games_data:
            game_details = recommender.games_data[item_id]
            rec.update({k: game_details[k] for k in attributes_to_include if k in game_details})
        
        result.append(rec)
    
    return result

# Background task to save the model
async def save_model_in_background(recommender: GameRecommender):
    async with model_lock:
        recommender.save(FUNK_SVD_MODEL_DIR)

# Run with: uvicorn src.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
