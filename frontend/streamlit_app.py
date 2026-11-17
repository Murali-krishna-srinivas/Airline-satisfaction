import os
import streamlit as st
import requests


def get_api_url():
    # 1) Streamlit Cloud secret  2) environment variable  3) local default
    try:
        return st.secrets["API_URL"].rstrip("/")
    except Exception:
        return os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


API_URL = get_api_url()

st.title("Airline Passenger Satisfaction App")
st.write("Enter passenger and flight details to predict satisfaction")

# Input widgets
st.header("Passenger and Flight Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    age = st.number_input("Age", 7, 85, 30)
    flight_distance = st.number_input("Flight Distance", 31, 5000, 800)

with col2:
    type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    departure_delay = st.number_input("Departure Delay in Minutes", 0, 1600, 0)
    arrival_delay = st.number_input("Arrival Delay in Minutes", 0, 1600, 0)

st.header("Service Ratings")
st.caption("0 = not applicable, 1 = very poor, 5 = excellent")

left, right = st.columns(2)

with left:
    inflight_wifi_service = st.slider("Inflight wifi service", 0, 5, 3)
    ease_of_online_booking = st.slider("Ease of Online booking", 0, 5, 3)
    food_and_drink = st.slider("Food and drink", 0, 5, 3)
    seat_comfort = st.slider("Seat comfort", 0, 5, 3)
    on_board_service = st.slider("On-board service", 0, 5, 3)
    baggage_handling = st.slider("Baggage handling", 0, 5, 3)
    inflight_service = st.slider("Inflight service", 0, 5, 3)

with right:
    departure_arrival_time_convenient = st.slider("Departure/Arrival time convenient", 0, 5, 3)
    gate_location = st.slider("Gate location", 0, 5, 3)
    online_boarding = st.slider("Online boarding", 0, 5, 3)
    inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 3)
    leg_room_service = st.slider("Leg room service", 0, 5, 3)
    checkin_service = st.slider("Checkin service", 0, 5, 3)
    cleanliness = st.slider("Cleanliness", 0, 5, 3)

# Prediction
if st.button("Predict Satisfaction"):

    data = {
        "gender": gender,
        "customer_type": customer_type,
        "age": age,
        "type_of_travel": type_of_travel,
        "travel_class": travel_class,
        "flight_distance": flight_distance,
        "inflight_wifi_service": inflight_wifi_service,
        "departure_arrival_time_convenient": departure_arrival_time_convenient,
        "ease_of_online_booking": ease_of_online_booking,
        "gate_location": gate_location,
        "food_and_drink": food_and_drink,
        "online_boarding": online_boarding,
        "seat_comfort": seat_comfort,
        "inflight_entertainment": inflight_entertainment,
        "on_board_service": on_board_service,
        "leg_room_service": leg_room_service,
        "baggage_handling": baggage_handling,
        "checkin_service": checkin_service,
        "inflight_service": inflight_service,
        "cleanliness": cleanliness,
        "departure_delay": departure_delay,
        "arrival_delay": arrival_delay
    }

    try:
        with st.spinner("Predicting... (the server may take up to a minute to wake up)"):
            response = requests.post(f"{API_URL}/predict", json=data, timeout=90)

        if response.status_code == 200:

            result = response.json()

            st.subheader("Result")

            if result["prediction"] == 1:
                st.success(
                    f'{result["result"]}\n\n'
                    f'Satisfaction Probability: {result["probability"]}%'
                )
            else:
                st.error(
                    f'{result["result"]}\n\n'
                    f'Satisfaction Probability: {result["probability"]}%'
                )

            st.progress(result["probability"] / 100)

        else:
            st.error("Prediction failed")

    except requests.exceptions.Timeout:
        st.error("The server is waking up. Please click Predict again in a minute.")
    except requests.exceptions.ConnectionError:
        st.error("FastAPI server is not running.")
