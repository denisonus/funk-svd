from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum

# Define data models for API
class Rating(BaseModel):
    BGGId: int = Field(..., description="Board Game Geek ID of the game")
    Rating: float = Field(..., ge=1.0, le=10.0, description="User rating from 1.0 to 10.0")

class UserRatings(BaseModel):
    ratings: List[Rating] = Field(description="List of game ratings")
    UserId: Optional[int] = Field(None, description="User ID (optional, will be assigned if not provided)")

class RecommendationResponse(BaseModel):
    BGGId: int
    PredictedRating: float
    Name: Optional[str] = None
    YearPublished: Optional[int] = None
    Description: Optional[str] = None
    ImageURL: Optional[str] = None
    
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to get recommendations for")
    num_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations to return")
    include_details: bool = Field(True, description="Whether to include game details in the response")
    
class GameDetails(BaseModel):
    BGGId: int
    Name: str
    YearPublished: Optional[int] = None
    Description: Optional[str] = None
    ImageURL: Optional[str] = None
    AverageRating: Optional[float] = None
    
class PopularRecommendationRequest(BaseModel):
    num_recommendations: int = Field(10, ge=1, le=100, description="Number of popular games to return")
    include_details: bool = Field(True, description="Whether to include game details in the response")

# API status response
class StatusResponse(BaseModel):
    status: str
    message: str
    model_info: Dict[str, Any]

# Rating update response
class RatingUpdateResponse(BaseModel):
    success: bool
    user_id: int
    is_new_user: bool
    num_ratings_processed: int
    
# Define category for sort options
class SortOption(str, Enum):
    POPULARITY = "popularity"
    RATING = "rating"
    YEAR = "year"
    NAME = "name"
    
# Games list request
class GamesListRequest(BaseModel):
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: SortOption = Field(SortOption.POPULARITY, description="Sort field")
    sort_desc: bool = Field(True, description="Sort descending")
    name_filter: Optional[str] = Field(None, description="Filter by game name")
