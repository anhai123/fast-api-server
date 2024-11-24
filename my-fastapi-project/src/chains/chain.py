from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from models.job_rag_query import JobQueryInput, JobQueryInput
from typing import List, Dict
from models.job_rag_query import Message
from jobProcessingService import search_by_user_query
open_ai_key = os.environ['OPENAI_API_KEY']

contextualize_q_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. Include the 'Link to Company' and 'Requirements' "
    "in your response if available. If you don't know the answer, "
    "say that you don't know. Use three sentences maximum and keep the "
    "answer concise."
)


_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        # ("system", "{context}"),
        ("ai", "{jobQdrant}"),
        ("human", "{text}"),
    ]
)
_model = ChatOpenAI(model="gpt-4o-mini",
temperature = 0,
api_key = open_ai_key,
max_tokens = 100,
timeout = None,
max_retries = 2,)

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model

# Chain với OpenAI LLM
def create_langchain_response(message: JobQueryInput):
    print(message)
    history_chat = standardize_messages(message.messages)

    job_available = search_by_user_query(message.text)
    formatted_results_str = "\n\n".join([
    f"Score: {job['Score']}\n"
    f"Name: {job['Name']}\n"
    f"Application Deadline: {job['ApplicationDeadline']}\n"
    f"Link to Company: {job['LinkCompany']}\n"
    f"Description: {job['Description']}\n"
    f"Requirements: {job['Requirements']}\n"
    f"Benefits: {job['Benefits']}\n"
    f"Address: {job['Address']}\n"
    f"Working Hours: {job['WorkingHours']}\n"
    f"How to Apply: {job['HowToApply']}\n"
    f"Job ID: {job['JobId']}"
    for job in job_available
    ])


    # Truyền tin nhắn vào chain để nhận câu trả lời
    # response = chain.invoke({"text": f"{message.text}", "chat_history": history_chat, "context": source_knowledge})
    response = chain.invoke({"text": f"{message.text}", "chat_history": history_chat, "jobQdrant": formatted_results_str})
    return response



def standardize_messages(messages: List[Message]) -> List[Dict[str, str]]:
    standardized = []
    for message in messages:
        standardized.append({
            "role": "human",
            "content": message.question
        })
        standardized.append({
            "role": "assistant",
            "content": message.answer
        })
    return standardized

# Example usage
# messages = [
#     {
#         "question": "What is the capital of France?",
#         "answer": "Paris"
#     },
#     {
#         "question": "What is 2 + 2?",
#         "answer": "4"
#     }
# ]

# standardized_messages = standardize_messages(messages)
# print(standardized_messages)
