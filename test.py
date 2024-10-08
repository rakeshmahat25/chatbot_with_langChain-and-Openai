from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
import json
from sqlalchemy.exc import IntegrityError
import PyPDF2
from sqlalchemy import Column, Integer, String, Text, Date, create_engine
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Allow front-end on port 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Database setup
DATABASE_URL = "sqlite:///./cv_chatbot.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Define the database model for CV
class CV(Base):
    __tablename__ = "cv_data"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    email = Column(String(255), unique=True, nullable=True)
    phone_number = Column(String(50), nullable=True)
    linkedin_profile = Column(String(255), nullable=True)
    address = Column(String(255), nullable=True)
    professional_summary = Column(Text, nullable=True)
    job_title = Column(String(255), nullable=True)
    company_name = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    responsibilities = Column(Text, nullable=True)
    degree = Column(String(255), nullable=True)
    institution = Column(String(255), nullable=True)
    education_location = Column(String(255), nullable=True)
    graduation_date = Column(Date, nullable=True)
    field_of_study = Column(String(255), nullable=True)
    skills = Column(Text, nullable=True)
    certification_name = Column(String(255), nullable=True)
    issuing_organization = Column(String(255), nullable=True)
    certification_date = Column(Date, nullable=True)
    project_title = Column(String(255), nullable=True)
    project_description = Column(Text, nullable=True)
    project_date = Column(Date, nullable=True)
    technologies_used = Column(String(255), nullable=True)


Base.metadata.create_all(bind=engine)


def fetch_data(query):
    conn = sqlite3.connect('cv_chatbot.db')
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <body>
            <h1>Welcome to the CV Chatbot API!</h1>
            <p>Use the endpoints /chat and /upload_cv for interacting with the CV data.</p>
            <br>
            <br>
            <div id="chat-container">
        </div>
        </body>
    </html>
    """


@app.post("/upload_cv")
async def upload_cv(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Save the uploaded file
    upload_folder = 'static/temp'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text from the PDF
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    max_text_length = 3000  # Adjust as needed, but this is just an example
    text = text[:max_text_length]

    # Prepare the prompt for OpenAI
    prompt = f"Extract the following fields from the CV text:\n{text}\n\n" \
             "Please provide the following details in JSON format:\n" \
             "{ \"Name\": \"\", \"Email\": \"\", \"Phone_Number\": \"\", " \
             "\"LinkedIn_Profile\": \"\", \"Address\": \"\", \"Professional_Summary\": \"\", " \
             "\"Job_Title\": \"\", \"Company_Name\": \"\", \"Location\": \"\", " \
             "\"Start_Date\": \"\", \"End_Date\": \"\", \"Responsibilities\": \"\", " \
             "\"Degree\": \"\", \"Institution\": \"\", \"Education_Location\": \"\", " \
             "\"Graduation_Date\": \"\", \"Field_of_Study\": \"\", \"Skills\": \"\", " \
             "\"Certification_Name\": \"\", \"Issuing_Organization\": \"\", " \
             "\"Certification_Date\": \"\", \"Project_Title\": \"\", " \
             "\"Project_Description\": \"\", \"Project_Date\": \"\", \"Technologies_Used\": \"\"}"

    try:
        # Use the new completions API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000  # You can adjust the tokens as per your need
        )
        cv_data = json.loads(response['choices'][0]['message']['content'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting CV data: {str(e)}")

    def process_field(field_name, default=""):
        """Helper function to ensure the field is a string, and join lists if needed."""
        value = cv_data.get(field_name, default)

        if isinstance(value, list):  # If the value is a list, we need to join the elements
            return ', '.join(str(item) for item in value if item is not None)

        return str(value)

    # Prepare the data for insertion
    cv_data_values = (
        process_field("Name"),
        process_field("Email"),
        process_field("Phone_Number"),
        process_field("LinkedIn_Profile"),
        process_field("Address"),
        process_field("Professional_Summary"),
        process_field("Job_Title"),
        process_field("Company_Name"),
        process_field("Location"),
        process_field("Start_Date"),
        process_field("End_Date"),
        process_field("Responsibilities"),
        process_field("Degree"),
        process_field("Institution"),
        process_field("Education_Location"),
        process_field("Graduation_Date"),
        process_field("Field_of_Study"),
        process_field("Skills"),
        process_field("Certification_Name"),
        process_field("Issuing_Organization"),
        process_field("Certification_Date"),
        process_field("Project_Title"),
        process_field("Project_Description"),
        process_field("Project_Date"),
        process_field("Technologies_Used")
    )

    # Insert data into the database
    conn = sqlite3.connect('cv_chatbot.db')
    cursor = conn.cursor()
    try:
        cursor.execute(''' 
                INSERT INTO cv_data (
                    Name, Email, Phone_Number, LinkedIn_Profile, Address, 
                    Professional_Summary, Job_Title, Company_Name, Location, 
                    Start_Date, End_Date, Responsibilities, Degree, Institution, 
                    Education_Location, Graduation_Date, Field_of_Study, Skills, 
                    Certification_Name, Issuing_Organization, Certification_Date, 
                    Project_Title, Project_Description, Project_Date, Technologies_Used
                ) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', cv_data_values)
        conn.commit()
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Email already exists. Please use a different email.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving CV data: {str(e)}")
    finally:
        conn.close()

    return {"detail": "CV uploaded successfully."}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
