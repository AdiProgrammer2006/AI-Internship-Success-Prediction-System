from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Placement Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("../model/placement_model.pkl")

feature_names = (
    pd.read_csv("../data/preprocessed_data.csv")
    .drop(columns=["placement_status"])
    .columns
    .tolist()
)


class Student(BaseModel):
    age: int
    gender: int
    cgpa: float
    internships_count: int
    projects_count: int
    certifications_count: int
    coding_skill_score: float
    aptitude_score: float
    communication_skill_score: float
    logical_reasoning_score: float
    hackathons_participated: int
    github_repos: int
    linkedin_connections: int
    mock_interview_score: float
    attendance_percentage: float
    backlogs: int
    extracurricular_score: float
    leadership_score: float
    volunteer_experience: int
    sleep_hours: float
    study_hours_per_day: float

    branch_CSE: int
    branch_Civil: int
    branch_ECE: int
    branch_EEE: int
    branch_IT: int
    branch_Mechanical: int

    college_tier_Tier_1: int
    college_tier_Tier_2: int
    college_tier_Tier_3: int


@app.get("/")
def home():
    return {
        "message": "Placement Prediction API is running!"
    }


@app.post("/predict")
def predict(student: Student):
    input_df = pd.DataFrame([student.model_dump()])

    input_df.rename(columns={
        "college_tier_Tier_1": "college_tier_Tier 1",
        "college_tier_Tier_2": "college_tier_Tier 2",
        "college_tier_Tier_3": "college_tier_Tier 3",
    }, inplace=True)

    input_df = input_df[feature_names]

    prediction = int(model.predict(input_df)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = round(float(probabilities[prediction]) * 100, 2)

    return {
        "prediction": prediction,
        "placement_status": "Placed" if prediction == 1 else "Not Placed",
        "confidence": confidence
    }