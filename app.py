from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
from sqlalchemy import Column, Integer, String, Text, Date, create_engine
from langchain.llms import OpenAI
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


@app.post("/chat")
async def chat(message: str):
    """Chat endpoint to interact with CV data and return responses from OpenAI."""
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' query parameter")

    try:

        query = "SELECT * FROM cv_data"

        cv_data = fetch_data(query)

        # Prompt template for LangChain
        prompt_template = """
                You are an intelligent assistant. Given the following CV information: 
                {cv_data}
                Answer the user's query: {query}
                """

        # Initialize LangChain OpenAI model
        llm = OpenAI(temperature=0.5)

        # Define the prompt template
        prompt = PromptTemplate(template=prompt_template, input_variables=["cv_data", "query"])

        # Adjusting the access to tuple elements by index, not using named attributes
        cv_data_text = "\n".join([
            f"Name: {row[1]}, Email: {row[2]}, Phone: {row[3]}, LinkedIn: {row[4]}, Address: {row[5]}, "
            f"Summary: {row[6]}, Job Title: {row[7]}, Company: {row[8]}, Location: {row[9]}, "
            f"Start Date: {row[10]}, End Date: {row[11]}, Responsibilities: {row[12]}, Degree: {row[13]}, "
            f"Institution: {row[14]}, Education Location: {row[15]}, Graduation Date: {row[16]}, "
            f"Field of Study: {row[17]}, Skills: {row[18]}, Certification Name: {row[19]}, "
            f"Issuing Organization: {row[20]}, Certification Date: {row[21]}, Project Title: {row[22]}, "
            f"Project Description: {row[23]}, Project Date: {row[24]}, Technologies Used: {row[25]}"
            for row in cv_data
        ])

        formatted_prompt = prompt.format(cv_data=cv_data_text, query=message)

        response = llm(formatted_prompt)

        return {"response": response}

    except Exception as e:
        # Detailed logging for debugging
        raise HTTPException(status_code=500, detail=f"Error during chat operation: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
