from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from uvicorn import run as app_run
from typing import Optional

from src.pipeline.predict import VisaData, VisaClassifier
from src.pipeline.train import TrainPipeline
from src.constants import APP_HOST, APP_PORT


# Initialize FastAPI app
app = FastAPI()

# Serve static files with FastAPI
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

# CORS middleware configuration
origins = ["*"]
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.continent: Optional[str] = None
        self.education_of_employee: Optional[str] = None
        self.has_job_experience: Optional[str] = None
        self.requires_job_training: Optional[str] = None
        self.no_of_employees: Optional[str] = None
        self.region_of_employment: Optional[str] = None
        self.prevailing_wage: Optional[str] = None
        self.unit_of_wage: Optional[str] = None
        self.full_time_position: Optional[str] = None
        self.yr_of_estab: Optional[str] = None


    async def get_visa_data(self):
        form = await self.request.form()
        self.continent = form.get("continent")
        self.education_of_employee = form.get("education_of_employee")
        self.has_job_experience = form.get("has_job_experience")
        self.requires_job_training = form.get("requires_job_training")
        self.no_of_employees = form.get("no_of_employees")
        self.region_of_employment = form.get("region_of_employment")
        self.prevailing_wage = form.get("prevailing_wage")
        self.unit_of_wage = form.get("unit_of_wage")
        self.full_time_position = form.get("full_time_position")
        self.yr_of_estab = form.get("yr_of_estab")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "context": "???"})


@app.get("/train")
async def trigger_training_pipeline():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Train: Success")
    except Exception as e:
        return Response(f"Train: Error Occurred [{e}]")


@app.post("/predict")
async def predict_visa_status(request: Request):
    try:
        form = DataForm(request)
        await form.get_visa_data()
        visa_data = VisaData(continent=form.continent,
                             employee_education=form.education_of_employee,
                             has_job_experience=form.has_job_experience,
                             requires_job_training=form.requires_job_training,
                             no_of_employees=int(form.no_of_employees),
                             region_of_employment=form.region_of_employment,
                             prevailing_wage=float(form.prevailing_wage),
                             unit_of_wage=form.unit_of_wage,
                             full_time_position=form.full_time_position,
                             yr_of_estab=int(form.yr_of_estab))

        visa_df = visa_data.convert_to_dataframe()
        model = VisaClassifier()

        # Use in local deployment
        outcome = model.predict_local(dataframe=visa_df)[0]

        # Use in production (AWS S3)
        # outcome = model.predict_s3(dataframe=visa_df)[0]

        status = "Certified" if outcome == 1 else "Denied"
        # Return the predicted outcome as JSON response
        return {"context": status}
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
