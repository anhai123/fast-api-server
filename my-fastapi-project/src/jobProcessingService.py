from typing import Any, Dict, List
from openai import OpenAI
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import PointStruct
from qdrantConnection import qdrant_client, collection_name, vector_store, embedding_model
import uuid
import os
import json
from datetime import datetime

# Initialize OpenAI client
client = OpenAI()
client.api_key = os.environ['OPENAI_API_KEY']

def get_embedding(text, model="text-embedding-3-small"):
    """
    Generate an embedding for the given text using the specified model.

    :param text: The text to be embedded.
    :param model: The model to use for generating the embedding.
    :return: The embedding vector.
    """
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def insert_jobs_into_qdrant(jobs):
    """
    Insert job descriptions into the Qdrant collection.

    :param jobs: A list of job dictionaries containing job descriptions and metadata.
    """
    points = []
    for job in jobs:
        embedding = get_embedding(job['Description'], model=embedding_model)
        point = PointStruct(
            id=job['JobId'],
            vector=embedding[:1536],  # Adjust embedding dimension
            payload=job  # Stores metadata
        )
        points.append(point)

    # Batch insert
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Inserted {len(jobs)} jobs into the '{collection_name}' collection.")
    return "Inserted jobs successfully."

def get_existing_jobs():
    """
    Retrieve existing jobs from the Qdrant collection.

    :return: A set of existing job IDs.
    """
    existing_jobs = qdrant_client.scroll(
        collection_name=collection_name,
        limit=1000  # Adjust based on your data size
    )[0]  # Extract the list of records from the tuple

    current_date = datetime.now().date()
    outdated_jobs = []

    for record in existing_jobs:
        date_str = record.payload['ApplicationDeadline']
        for fmt in ('%Y-%m-%d', '%d/%m/%Y'):
            try:
                application_deadline = datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Date format for {date_str} is not supported.")
        if application_deadline < current_date:
            outdated_jobs.append(record.payload['JobId'])

    job_ids = [record.payload['JobId'] for record in existing_jobs]
    ids = [record.id for record in existing_jobs]

    print("Outdated job ids: ", outdated_jobs)
    print("Job ids: ", job_ids)
    print(" ids: ", ids)
    return set(job_ids)


def get_outdated_jobs():
    """
    Retrieve a list of outdated job IDs based on their application deadlines.
    This function fetches existing job records from a specified collection,
    checks their application deadlines, and identifies jobs that are outdated
    (i.e., their application deadlines have passed).
    Returns:
        list: A list of job IDs that are outdated.
    Raises:
        ValueError: If the date format of 'ApplicationDeadline' is not supported.
    Note:
        The function assumes that the 'ApplicationDeadline' field in the job records
        can be in either '%Y-%m-%d' or '%d/%m/%Y', '%Y-%m-%d %H:%M:%S' format.
    """
    existing_jobs = qdrant_client.scroll(
        collection_name=collection_name,
        limit=1000  # Adjust based on your data size
    )[0]  # Extract the list of records from the tuple

    current_date = datetime.now().date()
    outdated_jobs = []

    for record in existing_jobs:
        date_str = record.payload['ApplicationDeadline']
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S'):
            try:
                application_deadline = datetime.strptime(date_str, fmt).date()
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Date format for {date_str} is not supported.")
        if application_deadline < current_date:
            outdated_jobs.append(record.payload['JobId'])

    print("Outdated job ids: ", outdated_jobs)
    return outdated_jobs


def identify_and_remove_outdated_jobs():
    """
    Identify outdated jobs in the Qdrant collection.
    """
    outdated_job_ids = get_outdated_jobs()
    print("Outdated job ids: ", outdated_job_ids)
    if outdated_job_ids:
        # remove_outdated_jobs(outdated_job_ids)
        print(f"Removed {len(outdated_job_ids)} outdated jobs.")
    else:
        print("No outdated jobs to remove.")

def remove_outdated_jobs(outdated_job_ids):
    """
    Remove outdated jobs from the Qdrant collection.

    :param outdated_job_ids: A list of job IDs to be removed.
    """
    qdrant_client.delete(
        collection_name=collection_name,
        points_selector={
            "filter": {
                "must": [
                    {
                        "key": "JobId",
                        "match": {
                            "value": job_id
                        }
                    }
                    for job_id in outdated_job_ids
                ]
            }
        }
    )

def delete_all_jobs():
    """
    Deletes all points (jobs) from the specified Qdrant collection without deleting the collection.
    """
    try:
        # Retrieve all point IDs (scroll through points)
        points, _ = qdrant_client.scroll(collection_name=collection_name, limit=10000)  # Adjust the limit as needed
        point_ids = [point.id for point in points]  # Extract IDs from the points

        if point_ids:
            # Use the correct PointIdsList selector to delete points
            qdrant_client.delete(collection_name=collection_name, points_selector=qdrant_models.PointIdsList(points=point_ids))
            print(f"Deleting jobs with IDs: {point_ids}")
            print(f"All jobs have been successfully deleted from the '{collection_name}' collection.")
        else:
            print(f"No jobs found in the '{collection_name}' collection.")
    except Exception as e:
        print(f"An error occurred while deleting jobs: {e}")

def load_jobs_from_file(file_path):
    """
    Load jobs from a JSON file.

    :param file_path: The path to the JSON file containing job data.
    :return: A list of job dictionaries.
    """
    print("Loading jobs from file...")
    with open(file_path, 'r', encoding='utf-8') as file:
        jobs = json.load(file)
    return jobs

def search_jobs_by_metadata(filter_criteria):
    """
    Searches for jobs in the Qdrant collection based on the provided metadata filter criteria.

    :param filter_criteria: A dictionary containing the filter criteria for the search.
    :return: A list of jobs that match the filter criteria.
    """
    try:
        # Construct the filter query
        filter_query = {
            "must": [
                {
                    "key": key,
                    "match": {
                        "value": value
                    }
                }
                for key, value in filter_criteria.items()
            ]
        }

        # Perform the search
        search_result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_query,  # Use 'scroll_filter' instead of 'filter'
            limit=1000  # Adjust the limit as needed
        )[0]  # Extract the list of records from the tuple

        # Extract job payloads from the search result
        jobs = [record.payload for record in search_result]
        return jobs

    except Exception as e:
        print(f"An error occurred while searching for jobs: {e}")
        return []


def search_by_user_query(user_query):
    """
    Search for jobs in the Qdrant collection based on the user query.
    """
    # user_query = input("Enter your search query: ")
    # Perform the search based on the user query

    # Search by text similarity
    results = search_jobs(
        query=user_query,
        limit=1
    )
    formatted_results = []
    for result in results:
        job_data = result.payload
        score = result.score
        formatted_results.append({
            "Score": score,
            "Name": job_data.get("Name", ""),
            "ApplicationDeadline": job_data.get("ApplicationDeadline", ""),
            "LinkCompany": job_data.get("LinkCompany", ""),
            "Description": job_data.get("Description", ""),
            "Requirements": job_data.get("Requirements", ""),
            "Benefits": job_data.get("Benefits", ""),
            "Address": job_data.get("Address", ""),
            "WorkingHours": job_data.get("WorkingHours", ""),
            "HowToApply": job_data.get("HowToApply", ""),
            "JobId": job_data.get("JobId", "")
        })
    return formatted_results

def search_jobs(
    query: str,
    limit: int = 3,
    score_threshold: float = 0.8,
    max_depth: int = 3
) -> List[Dict[str, Any]]:
    try:
        query_vector = embedding_model.embed_query(query)
        all_results = []
        visited_ids = set()  # To avoid duplicate results
        current_query_vector = query_vector
        depth = 0

        while depth < max_depth:
            # Perform the search with the current query vector
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=current_query_vector,
                limit=limit
            )

            # Filter and add new results
            for result in results:
                job_id = result.payload.get("id")
                if job_id not in visited_ids and result.score >= score_threshold:
                    visited_ids.add(job_id)
                    all_results.append(result)

            # Generate a new query vector based on results (refinement step)
            if results:
                # Example: Aggregate descriptions or titles for query refinement
                context = " ".join([res.payload.get("title", "") + " " + res.payload.get("description", "") for res in results])
                current_query_vector = embedding_model.embed_query(context)
            else:
                break  # Stop if no more results are found

            depth += 1

        return all_results[:limit]  # Return top results up to the limit

    except Exception as e:
        print(f"Error during recursive job search: {str(e)}")
        return []



if __name__ == "__main__":
    # file_path = 'recruit_1_9.json'
    # jobs = load_jobs_from_file(file_path)
    # insert_jobs_into_qdrant(jobs)
    # get_existing_jobs()


    res = search_by_user_query("I want to quit job?")
    print (res)


    # delete_all_jobs()

    # filter_criteria = {
    #     "name": "Công ty cổ phần phần mềm ITSOL Holding",
    # }
    # matching_jobs = search_jobs_by_metadata(filter_criteria)
    # print("Matching jobs:", matching_jobs)
