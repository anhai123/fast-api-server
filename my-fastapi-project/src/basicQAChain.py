from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from jobProcessingService import search_by_user_query
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

# Retrieve OpenAI API key from environment variables
open_ai_key = os.environ['OPENAI_API_KEY']

# Initialize the ChatOpenAI model
_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=open_ai_key,
    max_tokens=100,
    timeout=None,
    max_retries=2,
)

# Define the prompt template
template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

### CONTEXT
{context}

### QUESTION
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Define the retrieval augmented QA chain
retrieval_augmented_qa_chain = (
    {"context": RunnableLambda(lambda x: search_by_user_query(x["question"])), "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | _model, "context": itemgetter("context")}
)



if __name__ == "__main__":
    question = "What is RAG?"

    # Invoke the retrieval augmented QA chain with the question
    result = retrieval_augmented_qa_chain.invoke({"question": question})

    # Print the result
    print("result")
    print(result)
