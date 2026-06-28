from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Placement Prediction API")

model = joblib.load("../model/placement_model.pkl")


class PlacementInput(BaseModel):
    Gender: int
    tenth_marks: float
    twelfth_marks: float
    Cgpa: float
    Internships: int
    Training: int
    Backlog_5th_sem: int
    Innovative_Project: float
    Communication_level: int
    Technical_Course: float

    board10_CBSE: bool
    board10_ICSE: bool
    board10_State: bool
    board10_WBBSE: bool

    board12_BSEB: bool
    board12_CBSE: bool
    board12_CISCE: bool
    board12_Diploma: bool
    board12_MSBTE_Diploma: bool
    board12_ISC: bool
    board12_ISE: bool
    board12_MSBTE: bool
    board12_Other_State: bool
    board12_WBBSE: bool
    board12_WBCHSE: bool

    Stream_Chemical_Engineering: bool
    Stream_Civil_Engineering: bool
    Stream_Computer_Science_and_Design: bool
    Stream_Computer_Science_and_Engineering: bool
    Stream_Computer_Science_in_AIML: bool
    Stream_Computer_Science_in_Data_Science: bool
    Stream_Electrical_Engineering: bool
    Stream_Electrical_and_Electronics_Engineering: bool
    Stream_Electronic_Engineering: bool
    Stream_Electronics_Engineering: bool
    Stream_Electronics_and_Communication_Engineering: bool
    Stream_Electronics_and_Communication_and_Engineeing: bool
    Stream_IMsc_Maths_and_Computing: bool
    Stream_Information_Technology: bool
    Stream_Mechanical_Engineering: bool
    Stream_Production_Engineering: bool


@app.get("/")
def home():
    return {"message": "Placement Prediction API"}


@app.post("/predict")
def predict(data: PlacementInput):

    df = pd.DataFrame([{
        "Gender": data.Gender,
        "10th marks": data.tenth_marks,
        "12th marks": data.twelfth_marks,
        "Cgpa": data.Cgpa,
        "Internships(Y/N)": data.Internships,
        "Training(Y/N)": data.Training,
        "Backlog in 5th sem": data.Backlog_5th_sem,
        "Innovative Project(Y/N)": data.Innovative_Project,
        "Communication level": data.Communication_level,
        "Technical Course(Y/N)": data.Technical_Course,

        "10th board_CBSE": data.board10_CBSE,
        "10th board_ICSE": data.board10_ICSE,
        "10th board_State Board": data.board10_State,
        "10th board_WBBSE": data.board10_WBBSE,

        "12th board_BSEB": data.board12_BSEB,
        "12th board_CBSE": data.board12_CBSE,
        "12th board_CISCE": data.board12_CISCE,
        "12th board_Diploma": data.board12_Diploma,
        "12th board_Diploma board - MSBTE": data.board12_MSBTE_Diploma,
        "12th board_ISC": data.board12_ISC,
        "12th board_ISE": data.board12_ISE,
        "12th board_MSBTE": data.board12_MSBTE,
        "12th board_Other state Board": data.board12_Other_State,
        "12th board_WBBSE": data.board12_WBBSE,
        "12th board_WBCHSE": data.board12_WBCHSE,

        "Stream_Chemical Engineering": data.Stream_Chemical_Engineering,
        "Stream_Civil Engineering": data.Stream_Civil_Engineering,
        "Stream_Computer Science and Design": data.Stream_Computer_Science_and_Design,
        "Stream_Computer Science and Engineering": data.Stream_Computer_Science_and_Engineering,
        "Stream_Computer Science in AIML": data.Stream_Computer_Science_in_AIML,
        "Stream_Computer Science in Data Science": data.Stream_Computer_Science_in_Data_Science,
        "Stream_Electrical Engineering": data.Stream_Electrical_Engineering,
        "Stream_Electrical and Electronics Engineering": data.Stream_Electrical_and_Electronics_Engineering,
        "Stream_Electronic Engineering": data.Stream_Electronic_Engineering,
        "Stream_Electronics Engineering": data.Stream_Electronics_Engineering,
        "Stream_Electronics and Communication Engineering": data.Stream_Electronics_and_Communication_Engineering,
        "Stream_Electronics and Communication and Engineeing": data.Stream_Electronics_and_Communication_and_Engineeing,
        "Stream_IMsc Maths and Computing": data.Stream_IMsc_Maths_and_Computing,
        "Stream_Information Technology": data.Stream_Information_Technology,
        "Stream_Mechanical Engineering": data.Stream_Mechanical_Engineering,
        "Stream_Production Engineering": data.Stream_Production_Engineering,
    }])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    return {
        "prediction": int(prediction),
        "placement": "Placed" if prediction == 1 else "Not Placed",
        "probability": {
            "Not Placed": round(float(probability[0]), 4),
            "Placed": round(float(probability[1]), 4)
        }
    }