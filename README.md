# Visa Approval Prediction

## Problem Definition
#### Context
Business communities in the United States (US) are facing high demand for human resources, but one of the constant challenges is identifying and attracting the right talent, which is perhaps the most important element in remaining competitive. Companies in the US look for hardworking, talented, and qualified individuals both locally as well as abroad.

The US Immigration and Nationality Act (INA) permits foreign workers to enter the country to work on either a temporary or permanent basis. The act also protects US workers against adverse impacts on their wages or working conditions by ensuring US employers' compliance with statutory requirements when hiring foreign workers to fill workforce shortages. The immigration programs are administered by the Office of Foreign Labor Certification (OFLC).

OFLC processes job certification applications for employers seeking to bring foreign workers into US and grants certifications in those cases where employers can demonstrate that there are insufficient US workers available to perform the work at wages that meet or exceed the wage paid for the occupation in the area of intended employment.

#### Objective
In FY2016, the OFLC processed 775,979 employer applications for 1,699,957 positions on temporary and permanent labor certifications. This was a 9% increase in the overall number of processed applications from the previous year. The process of reviewing every case is becoming a tedious task as the number of applicants is increasing every year.

The increasing number of applicants every year calls for a Machine Learning based solution that can help in shortlisting the candidates with higher chances of VISA approval. OFLC has hired the firm EasyVisa for data-driven solutions. You as a data scientist at EasyVisa have to analyze the data provided and, with the help of a classification model:
* Facilitate the process of visa approvals.
* Recommend a suitable profile for the applicants for whom the visa should be certified or denied based on the drivers that significantly influence the case status.

Successful implementation of this model promises to yield significant improvements for OFLC, including reduced processing time, operating costs, and administrative workload.

## Dataset
**Source**: [Kaggle - EasyVisa Dataset](https://www.kaggle.com/datasets/moro23/easyvisa-dataset)

**Description**:
- This dataset comprises a comprehensive set of attributes related to foreign employees who applied for a United States visa. These attributes encompass:
    - **Demographic Information:** Continent of origin, employee's education level, prevailing wage, and unit of wage.
    - **Employment History:** Prior work experience, requirement for job training, and whether the position is full-time.
    - **Employer Background:** Number of employees, year of establishment, and region of employment.
- The dataset also includes the final disposition of each application, categorized as `case_status`.

## Project Structure

```bash
visa-approval-prediction/
├── config/
│   ├── model.yaml
│   └── schema.yaml
├── notebook/
│   ├── data/
│   │   └── EasyVisa.csv
│   ├── data_drift_report.html
│   ├── eda.ipynb
│   └── model_training.ipynb
├── src/
│   ├── cloud_storage/
│   │   ├── __init__.py
│   │   └── aws_storage.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_evaluation.py
│   │   ├── model_pusher.py
│   │   └── model_trainer.py
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── aws_connection.py
│   │   └── mongo_db_connection.py
│   ├── constants/
│   │   └── __init__.py
│   ├── data_access/
│   │   ├── __init__.py
│   │   └── visa_data.py
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── artifact_entity.py
│   │   ├── config_entity.py
│   │   ├── estimator.py
│   │   └── s3_estimator.py
│   ├── exception/
│   │   └─── __init__.py
│   ├── logger/
│   │   └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_factory.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── utils/
│   │   └── __init__.py
│   └── __init__.py
├── static/
│   ├── css/
│   │   └── style.css
│   ├── images/
│   │   ├── favicon.ico
│   │   └── visa-approval.png
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
├── .gitignore
├── Dockerfile
├── README.md
├── app.py
├── requirements.txt
├── setup.py
└── template.py
```

## How to Run Locally
**1. Prerequisites**

You need to have Docker installed.

**2. Download the project files**

Clone this repository:
```bash
git clone https://github.com/wanyingng/visa-approval-prediction.git
```
Or, select "Download ZIP"

**3. Set up the environment**

Navigate to the project root directory:
```bash
cd visa-approval-prediction
```

Create a virtual environment and activate it:
```bash
python3.11 -m venv .venv
.\.venv\Scripts\activate
```

Install the project dependencies:
```bash
pip install -r requirements.txt
```

**4. Build the Dockerfile**

```bash
docker build -t visa-approval-service .
```

**5. Run the Docker image**

```bash
docker run -it --rm -p 9696:9696 visa-approval-service
```

**6. Access the FastAPI app through your web browser**

Enter `http://127.0.0.1:9696/` or `http://localhost:9696/` as the URL.
