from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
from typing import Annotated, Literal
import joblib

# Create FastAPI app
app = FastAPI()


# Health check (Render and uptime pingers can call this)

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "ok"}

# Load ML model
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "gradient_boosting_pipeline.pkl")

Rating = Annotated[int, Field(ge=0, le=5)]  # a rating from 0 to 5

# Input data structure
class PassengerInput(BaseModel):
    gender: Literal["Female", "Male"]
    customer_type: Literal["Loyal Customer", "disloyal Customer"]
    age: int = Field(ge=7, le=85)
    type_of_travel: Literal["Business travel", "Personal Travel"]
    travel_class: Literal["Business", "Eco", "Eco Plus"]
    flight_distance: int = Field(ge=31, le=5000)
    inflight_wifi_service: Rating
    departure_arrival_time_convenient: Rating
    ease_of_online_booking: Rating
    gate_location: Rating
    food_and_drink: Rating
    online_boarding: Rating
    seat_comfort: Rating
    inflight_entertainment: Rating
    on_board_service: Rating
    leg_room_service: Rating
    baggage_handling: Rating
    checkin_service: Rating
    inflight_service: Rating
    cleanliness: Rating
    departure_delay: int = Field(ge=0, le=1600)
    arrival_delay: int = Field(ge=0, le=1600)


# Same codes that LabelEncoder gave during training
GENDER = {"Female": 0, "Male": 1}
CUSTOMER = {"Loyal Customer": 0, "disloyal Customer": 1}
TRAVEL = {"Business travel": 0, "Personal Travel": 1}
CLASS = {"Business": 0, "Eco": 1, "Eco Plus": 2}


# Prediction API
@app.post("/predict")
def predict(data: PassengerInput):

    ratings = [
        data.inflight_wifi_service,
        data.departure_arrival_time_convenient,
        data.ease_of_online_booking,
        data.gate_location,
        data.food_and_drink,
        data.online_boarding,
        data.seat_comfort,
        data.inflight_entertainment,
        data.on_board_service,
        data.leg_room_service,
        data.baggage_handling,
        data.checkin_service,
        data.inflight_service,
        data.cleanliness,
    ]

    # Extra feature created during training
    average_rating = sum(ratings) / len(ratings)

    # One row, columns named and ordered exactly like X_train
    features = pd.DataFrame([[
        GENDER[data.gender],
        CUSTOMER[data.customer_type],
        data.age,
        TRAVEL[data.type_of_travel],
        CLASS[data.travel_class],
        data.flight_distance,
        *ratings,
        data.departure_delay,
        data.arrival_delay,
        average_rating,
    ]], columns=[
        "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance",
        "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink",
        "Online boarding", "Seat comfort", "Inflight entertainment",
        "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Inflight service", "Cleanliness",
        "Departure Delay in Minutes", "Arrival Delay in Minutes",
        "Average Service Rating",
    ])

    prediction = int(model.predict(features)[0])

    probability = float(model.predict_proba(features)[0][1] * 100)

    result = "Satisfied" if prediction == 1 else "Neutral or Dissatisfied"

    return {
        "prediction": prediction,
        "result": result,
        "probability": round(probability, 1)
    }