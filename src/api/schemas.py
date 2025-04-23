from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class Rating(BaseModel):
    BGGId: int = Field(..., description="Board Game Geek ID of the game")
    Rating: float = Field(..., ge=1.0, le=10.0, description="User rating from 1.0 to 10.0")

class UserRatings(BaseModel):
    ratings: List[Rating]
    UserId: Optional[int] = None

class RecommendationResponse(BaseModel):
    BGGId: int
    PredictedRating: float
    Name: Optional[str] = None
    YearPublished: Optional[int] = None
    Description: Optional[str] = None
    ImageURL: Optional[str] = None
    
class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = Field(10, ge=1, le=100)
    include_details: bool = True
    
class PopularRecommendationRequest(BaseModel):
    num_recommendations: int = Field(10, ge=1, le=100)
    include_details: bool = True

class StatusResponse(BaseModel):
    status: str
    message: str
    model_info: Dict[str, Any]

class RatingUpdateResponse(BaseModel):
    success: bool
    user_id: int
    is_new_user: bool
    num_ratings_processed: int
