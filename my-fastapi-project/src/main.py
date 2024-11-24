#from agents.hospital_rag_agent import hospital_rag_agent_executor
from fastapi import FastAPI,BackgroundTasks
from models.job import Job
from jobProcessingService import get_existing_jobs, identify_and_remove_outdated_jobs, insert_jobs_into_qdrant

from models.job_rag_query import JobQueryInput, JobQueryInput
from utils.async_utils import async_retry
from chains.chain import create_langchain_response
from fastapi.middleware.cors import CORSMiddleware

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from contextlib import asynccontextmanager
from CrawlDataBypassCapchaWithChromiumPage import beginCrawlData
from typing import List

# Create and configure the scheduler
scheduler = BackgroundScheduler()

#for background job
# Function to be run daily at 12 PM
def daily_task_background_remove_outdate_job():
    print(f"Task running at {datetime.now()}")
    identify_and_remove_outdated_jobs()
    # Add your job logic here (e.g., crawling logic)

def daily_task_background_crawl_data():
    print(f"Task running at {datetime.now()}")
    beginCrawlData()
    # Add your job logic here (e.g., crawling logic)



# Add the job to the scheduler (daily at 12 PM)
scheduler.add_job(daily_task_background_remove_outdate_job, CronTrigger(hour=4, minute=27))
scheduler.add_job(daily_task_background_crawl_data, CronTrigger(hour=4, minute=56))

# Define lifespan event handler using asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the scheduler on application startup
    scheduler.start()
    print("Scheduler started.")

    # Yield control to the application
    yield

    # Shutdown the scheduler on application shutdown
    scheduler.shutdown()
    print("Scheduler shutdown.")

# Set the lifespan handler for the FastAPI app
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows only http://localhost:3000
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



# Function to be run daily at 12 PM
# async def daily_task():
#     print(f"Task running at {datetime.now()}")
#     beginCrawlData()
#     # Add your job logic here (e.g., crawling logic)

# # Example route to show the app is running
# @app.get("/")
# async def read_root(background_tasks: BackgroundTasks):
#     try:
#         background_tasks.add_task(daily_task)
#     except Exception as e:
#         return {"error": str(e)}
#     return {"message": "Crawler is running in the background"}

@app.post("/process-message")
async def process_message(user_message: JobQueryInput):
    # Nhận tin nhắn từ người dùng
    message = user_message.text

    # Truyền vào Langchain để xử lý bằng OpenAI
    response = create_langchain_response(user_message)

    # Extract only the 'content' field from the response
    #response_content = response.get("content", "")

    # Return the response with only the 'content' field
    return {"response": response}


@app.post("/insert-job")
def insert_job(jobs: List[Job]):
    response = insert_jobs_into_qdrant(jobs)
    return {"response": response}

@app.get("/jobs-in-qdrant-database")
def get_jobs_in_qdrant_database():
    # Assuming you have a function to fetch jobs from Qdrant
    response = get_existing_jobs()
    return {"response": response}
