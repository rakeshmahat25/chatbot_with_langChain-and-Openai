from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
from sqlalchemy import Column, Integer, String, Text, Date, create_engine
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from sqlalchemy import text
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
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
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        print("Opening database session...")  # Debugging statement
        yield db
    finally:
        db.close()
        print("Closing database session...")


# Define SQLAlchemy model for CV data
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


# Create database tables
Base.metadata.create_all(bind=engine)

# Global dictionary to store conversation history per user session
conversation_history = {}

# Dictionary to track the selected candidate during the conversation
selected_candidate = None  # Track the currently selected candidate


def fetch_data(query: str, db: Session = None, page: int = 1, page_size: int = 100):
    """
    Fetch data from the database using SQLAlchemy ORM.
    Uses a raw SQL query or ORM models.

    Parameters:
    - query (str): The SQL query to execute.
    - db (Session): SQLAlchemy Session object.

    Returns:
    - result (list of tuples): The query results.
    """
    try:
        offset = (page - 1) * page_size  # Calculate the offset based on the page
        query_with_pagination = f"{query} LIMIT {page_size} OFFSET {offset}"

        # Execute the paginated query
        result = db.execute(text(query_with_pagination)).fetchall()

        # Debugging statement to inspect the fetched data
        print("Fetched data:", result)

        return result

    except Exception as e:
        print(f"Error occurred while fetching data: {e}")
        return []




# # Fetch data from the database using SQLAlchemy
# def fetch_data(query: str, db: Session = None, page: int = 1, page_size: int = 100):
#     """
#     Fetch data from the database using SQLAlchemy ORM.
#     Uses a raw SQL query or ORM models.
#
#     Parameters:
#     - query (str): The SQL query to execute.
#     - db (Session): SQLAlchemy Session object.
#
#     Returns:
#     - result (list of tuples): The query results.
#     """
#     try:
#         offset = (page - 1) * page_size  # Calculate the offset based on the page
#         query_with_pagination = f"{query} LIMIT {page_size} OFFSET {offset}"
#
#         # Execute the paginated query
#         result = db.execute(text(query_with_pagination)).fetchall()
#         return result
#
#     except Exception as e:
#         print(f"Error occurred while fetching data: {e}")
#         return []


# @app.post("/chat")
# async def chat(message: str):
#     """Chat endpoint to interact with CV data and return responses from OpenAI."""
#     global selected_candidate  # To access the global candidate variable
#
#     if not message:
#         raise HTTPException(status_code=400, detail="Missing 'message' query parameter")
#
#     # Unique session ID (could be a user identifier or session token, for now it's using 'default')
#     session_id = "default"
#
#     # If a new session starts, initialize an empty conversation history for that session
#     if session_id not in conversation_history:
#         conversation_history[session_id] = []
#
#     try:
#         # Ensure we create a valid DB session using the context manager
#         with get_db() as db:  # Use the context manager here
#             # Start by building the base query
#             query = "SELECT * FROM cv_data"
#
#             # Fetch CV data from database
#             cv_data = fetch_data(query, db)  # Pass the DB session
#
#             if not cv_data:
#                 return {"response": "No matching CV data found."}
#
#             # Dynamically build CV data text based on the database schema
#             cv_data_text = []
#             for row in cv_data:
#                 cv_text = []
#                 # Access each column value by index (or use column name if needed)
#                 cv_dict = {column.name: row[column.index] for column in CV.__table__.columns}
#
#                 # Dynamically decide which sections of the CV to include based on the query
#                 if "work experience" in message.lower():
#                     if cv_dict.get("job_title") and cv_dict.get("company_name") and cv_dict.get("responsibilities"):
#                         cv_text.append(f"Job Title: {cv_dict['job_title']}")
#                         cv_text.append(f"Company: {cv_dict['company_name']}")
#                         cv_text.append(f"Responsibilities: {cv_dict['responsibilities']}")
#
#                 if "education" in message.lower():
#                     if cv_dict.get("degree") and cv_dict.get("institution") and cv_dict.get("graduation_date"):
#                         cv_text.append(f"Degree: {cv_dict['degree']}")
#                         cv_text.append(f"Institution: {cv_dict['institution']}")
#                         cv_text.append(f"Graduation Date: {cv_dict['graduation_date']}")
#
#                 if "skills" in message.lower():
#                     if cv_dict.get("skills"):
#                         cv_text.append(f"Skills: {cv_dict['skills']}")
#
#                 if cv_text:
#                     cv_data_text.append("\n".join(cv_text))
#
#             # If we have a selected candidate, use them for the query context
#             if selected_candidate:
#                 # Add candidate details to the prompt
#                 cv_data_text = [f"Name: {selected_candidate['name']}"]  # Focus on the selected candidate
#                 if "education" in message.lower() and selected_candidate.get("degree"):
#                     cv_data_text.append(f"Degree: {selected_candidate['degree']}")
#                 if "skills" in message.lower() and selected_candidate.get("skills"):
#                     cv_data_text.append(f"Skills: {selected_candidate['skills']}")
#
#             # Update conversation history with the current message
#             conversation_history[session_id].append(f"User: {message}")
#
#             # Build conversation context from history (limit to last 5 exchanges to avoid long prompts)
#             conversation_context = "\n".join(conversation_history[session_id][-10:])  # Limit to the last 5 interactions
#
#             # Prepare the prompt template with dynamic CV data
#             prompt_template = """
#             You are an intelligent assistant. Based on the CV data provided below, answer the user's query.
#             CV Data:
#             {cv_data}
#             Previous conversation context:
#             {conversation_context}
#             Query: {query}
#             Answer:
#             """
#
#             # Initialize LangChain OpenAI model
#             llm = OpenAI(temperature=0.3)
#
#             # Define the prompt template
#             prompt = PromptTemplate(template=prompt_template, input_variables=["cv_data", "conversation_context", "query"])
#
#             # Format the prompt with the relevant CV data and conversation context
#             formatted_prompt = prompt.format(cv_data="\n\n".join(cv_data_text), conversation_context=conversation_context, query=message)
#
#             # Get response from OpenAI model
#             response = llm(formatted_prompt)
#
#             # Update conversation history with the new response
#             conversation_history[session_id].append(f"Assistant: {response}")
#
#             return {"response": response}
#
#     except Exception as e:
#         # Detailed logging for debugging
#         raise HTTPException(status_code=500, detail=f"Error during chat operation: {str(e)}")

@app.post("/chat")
async def chat(message: str):
    """Chat endpoint to interact with CV data and return responses from OpenAI."""
    global selected_candidate  # To access the global candidate variable

    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' query parameter")

    # Unique session ID (could be a user identifier or session token, for now it's using 'default')
    session_id = "default"

    # If a new session starts, initialize an empty conversation history for that session
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    try:
        # Ensure we create a valid DB session using the context manager
        with get_db() as db:  # Use the context manager here
            # Start by building the base query
            query = "SELECT * FROM cv_data"

            # Fetch CV data from database
            cv_data = fetch_data(query, db)  # Pass the DB session

            if not cv_data:
                return {"response": "No matching CV data found."}

            # Dynamically build CV data text based on the database schema
            cv_data_text = []
            for row in cv_data:
                cv_text = []
                # Accessing columns by index (position) in the tuple
                # Assuming your query returns rows as (id, name, email, ...) based on the cv_data schema
                cv_dict = {
                    "id": row[0],
                    "name": row[1],
                    "email": row[2],
                    "phone_number": row[3],
                    "linkedin_profile": row[4],
                    "address": row[5],
                    "professional_summary": row[6],
                    "job_title": row[7],
                    "company_name": row[8],
                    "location": row[9],
                    "start_date": row[10],
                    "end_date": row[11],
                    "responsibilities": row[12],
                    "degree": row[13],
                    "institution": row[14],
                    "education_location": row[15],
                    "graduation_date": row[16],
                    "field_of_study": row[17],
                    "skills": row[18],
                    "certification_name": row[19],
                    "issuing_organization": row[20],
                    "certification_date": row[21],
                    "project_title": row[22],
                    "project_description": row[23],
                    "project_date": row[24],
                    "technologies_used": row[25]
                }

                # Dynamically decide which sections of the CV to include based on the query
                if "work experience" in message.lower():
                    if cv_dict.get("job_title") and cv_dict.get("company_name") and cv_dict.get("responsibilities"):
                        cv_text.append(f"Job Title: {cv_dict['job_title']}")
                        cv_text.append(f"Company: {cv_dict['company_name']}")
                        cv_text.append(f"Responsibilities: {cv_dict['responsibilities']}")

                if "education" in message.lower():
                    if cv_dict.get("degree") and cv_dict.get("institution") and cv_dict.get("graduation_date"):
                        cv_text.append(f"Degree: {cv_dict['degree']}")
                        cv_text.append(f"Institution: {cv_dict['institution']}")
                        cv_text.append(f"Graduation Date: {cv_dict['graduation_date']}")

                if "skills" in message.lower():
                    if cv_dict.get("skills"):
                        cv_text.append(f"Skills: {cv_dict['skills']}")

                if cv_text:
                    cv_data_text.append("\n".join(cv_text))

            # If we have a selected candidate, use them for the query context
            if selected_candidate:
                # Add candidate details to the prompt
                cv_data_text = [f"Name: {selected_candidate['name']}"]  # Focus on the selected candidate
                if "education" in message.lower() and selected_candidate.get("degree"):
                    cv_data_text.append(f"Degree: {selected_candidate['degree']}")
                if "skills" in message.lower() and selected_candidate.get("skills"):
                    cv_data_text.append(f"Skills: {selected_candidate['skills']}")

            # Update conversation history with the current message
            conversation_history[session_id].append(f"User: {message}")

            # Build conversation context from history (limit to last 5 exchanges to avoid long prompts)
            conversation_context = "\n".join(conversation_history[session_id][-10:])  # Limit to the last 5 interactions

            # Prepare the prompt template with dynamic CV data
            prompt_template = """
            You are an intelligent assistant. Based on the CV data provided below, answer the user's query.
            CV Data: 
            {cv_data}
            Previous conversation context: 
            {conversation_context}
            Query: {query}
            Answer:
            """

            # Initialize LangChain OpenAI model
            llm = OpenAI(temperature=0.3)

            # Define the prompt template
            prompt = PromptTemplate(template=prompt_template, input_variables=["cv_data", "conversation_context", "query"])

            # Format the prompt with the relevant CV data and conversation context
            formatted_prompt = prompt.format(cv_data="\n\n".join(cv_data_text), conversation_context=conversation_context, query=message)

            # Get response from OpenAI model
            response = llm(formatted_prompt)

            # Update conversation history with the new response
            conversation_history[session_id].append(f"Assistant: {response}")

            return {"response": response}

    except Exception as e:
        # Detailed logging for debugging
        raise HTTPException(status_code=500, detail=f"Error during chat operation: {str(e)}")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
