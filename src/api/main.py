from asyncio import Lock
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends

from schemas import *
from src.config.settings import FUNK_SVD_MODEL_DIR
from src.data.load_dataset import get_train_data, get_games_data, get_test_data
from src.models.recommender import GameRecommender

instance = None
model_lock = Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global instance
    if instance is None:
        train_data = get_train_data()
        test_data = get_test_data()
        games_data = get_games_data()
        instance = GameRecommender(train_data, test_data, games_data)
        instance.load(FUNK_SVD_MODEL_DIR)

    yield

    if instance is not None:
        instance.save(FUNK_SVD_MODEL_DIR)


app = FastAPI(
    title="Game Recommender API",
    description="API for board game recommendations using FunkSVD algorithm",
    version="1.0.0",
    lifespan=lifespan
)


async def get_recommender():
    global instance
    if instance is None:
        raise HTTPException(
            status_code=500,
            detail="Recommender models not initialized. The service may still be starting up or encountered an initialization error."
        )

    return instance


@app.get("/", response_model=StatusResponse)
async def root(recommender: GameRecommender = Depends(get_recommender)):
    """Get API status and models information"""
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


@app.post("/ratings", response_model=RatingUpdateResponse)
async def add_ratings(
        ratings_data: UserRatings,
        background_tasks: BackgroundTasks,
        recommender: GameRecommender = Depends(get_recommender)
):
    """Add new ratings from a user and update the models"""
    ratings_list = [rating.model_dump() for rating in ratings_data.ratings]

    async with model_lock:
        success, user_id, is_new_user = recommender.add_ratings(
            ratings=ratings_list,
            user_id=ratings_data.UserId
        )

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

        if request.include_details and recommender.games_data and item_id in recommender.games_data:
            game_details = recommender.games_data[item_id]
            rec.update({k: game_details[k] for k in attributes_to_include if k in game_details})

        result.append(rec)

    return result


async def save_model_in_background(recommender: GameRecommender):
    async with model_lock:
        recommender.save(FUNK_SVD_MODEL_DIR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="localhost", port=8000, reload=True)
