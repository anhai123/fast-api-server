from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from finalResultEvaluation import evaluate_final_result
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from basicQAChain import retrieval_augmented_qa_chain
import json
open_ai_key = os.environ['OPENAI_API_KEY']
# Define the response schema for question generation
question_schema = ResponseSchema(
    name="question",
    description="a question about the context."
)

question_response_schemas = [question_schema]
question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
format_instructions = question_output_parser.get_format_instructions()

# Initialize the ChatOpenAI model for question generation
question_generation_llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0,
    api_key=open_ai_key,
    max_tokens=100,
    timeout=None,
    max_retries=2,
)

# Define a bare prompt template
bare_prompt_template = "{content}"
bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)


qa_template = """\
You are an AI Assistant helping a job seeker understand job postings and companies they are interested in applying to. For each job posting, generate a specific question that will help the job seeker gain a deeper understanding of the job role and the company. Avoid generic or general questions.

Your goal is to help the job seeker make an informed decision by asking insightful and relevant questions about:
- Job responsibilities and expectations
- Company culture and values
- Career growth opportunities
- Compensation and benefits
- Work-life balance
- Technical requirements and tools used
- Team structure and dynamics
- Application process and timelines

Format the output as JSON with the following keys:
question

context: {context}
"""

fake_context_for_generate_questions_and_groundtruth = [{
        'Score': 0.681165,
        'Name': 'Công ty TNHH Shinhan DS Việt Nam',
        'ApplicationDeadline': '30/11/2024',
        'LinkCompany': 'https://www.topcv.vn/viec-lam/software-developer-c-c/1505677.html?ta_source=JobSearchList_LinkDetail&u_sr_id=Szls65CkZkgRpYppK3m7LtwVuKQEv2cUdIyD1ljw_1730848215',
        'Description': "Shinhan DS Vietnam is established to be in charge of the ICT synergy of Shinhan Financial Group in Southeast Asia.\nSDSer's top priority is accelerating the client's business growth by strengthening their ICT systems.\nCollaborate closely with clients and IT professionals in the analysis, development, and testing of banking features & functions (web)\nWrite efficient source code to program complete applications.\nMaintain, modify or create new applications based on requirements.\nIdentify errors, and bugs, and devise solutions to these problems.\nAttend business (banking, stock/ securities & finance), and technical training from Korea's top experts.",
        'Requirements': 'From 3 years of experience in Software Development/Embedded Development\nProficiency in C/C++ based on Unix, and Linux.\nFamiliar with SQL/Oracle.\nGood to have: ProC and Finance & Banking field.\nGood problem-solving & communication skills.\nDetail-oriented, proactive, and fast learner.',
        'Benefits': "Our company provides an opportunity to grow as an expert Banking IT in Vietnam\nYou can learn about Banking concepts and know-how from Korea specialized developers\nIn addition, We have plans to expand our company to Indonesia, Myanmar, Philippines\nWe plan to offer training for outstanding employees in Korea. \nLet's Excite you more with our Brilliant perks:\nSalary: You will be happy + Pass-Probation Bonus\nPerformance Bonus: twice a year.\n12 days of annual leaves\nPersonal Allowances for birthdays, marriage, new babies, etc.\nTeam monthly allowance.\nSpecial loan offers and fee waivers from Shinhan Bank.\nCompany Trips, annual/monthly/weekly activities and events.\nHealth check once a year and premium healthcare program. \nEducation Programs and overseas training opportunities.",
        'Address': '- Hồ Chí Minh: The Mett, Thủ Đức',
        'WorkingHours': 'Thứ 2 - Thứ 6 (từ 08:00 đến 17:00)',
        'HowToApply': 'Ứng viên nộp hồ sơ trực tuyến bằng cách bấm Ứng tuyển ngay dưới đây.',
        'JobId': '881e3129-1059-4c5c-820a-dfc6cc732aae'
    }, {
        'Score': 0.6756177,
        'Name': 'CÔNG TY TNHH ISB VIỆT NAM',
        'ApplicationDeadline': '30/11/2024',
        'LinkCompany': 'https://www.topcv.vn/viec-lam/intern-frontend-dev-japanese-at-least-n4-tai-ho-chi-minh-thu-nhap-6-9-trieu-gross/1496246.html?ta_source=JobSearchList_LinkDetail&u_sr_id=Szls65CkZkgRpYppK3m7LtwVuKQEv2cUdIyD1ljw_1730848215',
        'Description': 'Collaborate with senior developers to design and develop user-friendly web interfaces. \nWrite clean, efficient, and well-structured code adhering to best practices and coding standards. \nUtilize JavaScript, jQuery, HTML, and CSS to create visually appealing and responsive web applications. \nIntegrate with RESTful APIs to fetch and manipulate data. \nConduct thorough debugging and utilize browser developer tools effectively. \nLearn and adapt to new technologies and frameworks as needed. \nParticipate in code reviews and provide constructive feedback.',
        'Requirements': 'Final year student or recent graduate majoring in Information Technology.\nFull-time internship and work full time after internship.\nStrong understanding of Object-Oriented Programming (OOP) principles.\nExperience in JavaScript, jQuery, HTML, and CSS.\nUnderstanding of RESTful API concepts and integration.\nStrong debugging and problem-solving skills.\nKnowledge of version control systems (Git, SVN).\nAbility to learn and adapt to new technologies quickly.\nGood communication and interpersonal skills.\nA passion for learning and growing as a developer.\nJapanese (N4 level or higher)',
        'Benefits': 'Attractive Intern salary.\nMotorbike or bus support allowance.\nHave a chance to be official member of ISB Vietnam.\nProfessional and challenge working environment.\nJapanese culture and working atmosphere discovery.',
        'Address': '- Hồ Chí Minh: E Town 2, 364 Cộng Hòa Phường 13, Tân Bình',
        'WorkingHours': 'Thứ 2 - Thứ 6 (từ 08:00 đến 17:00)',
        'HowToApply': 'Ứng viên nộp hồ sơ trực tuyến bằng cách bấm Ứng tuyển ngay dưới đây.',
        'JobId': '5925126c-b29a-4e71-ab1e-d9d8fef8b99e'
    }]

prompt_template = ChatPromptTemplate.from_template(template=qa_template)

messages = prompt_template.format_messages(
    context=fake_context_for_generate_questions_and_groundtruth,
    format_instructions=format_instructions
)

### Generate questions and context
question_generation_chain = bare_template | question_generation_llm

qac_triples = []

#create questions base on context
for text in tqdm(fake_context_for_generate_questions_and_groundtruth):
    messages = prompt_template.format_messages(
        context=text,
        format_instructions=format_instructions
    )
    response = question_generation_chain.invoke({"content" : messages})
    try:
        output_dict = question_output_parser.parse(response.content)
    except Exception as e:
        continue
    output_dict["context"] = text
    qac_triples.append(output_dict)

# {
#     'question': 'What opportunities for growth and advancement are available within the company?',
#     'context': {
#         'Score': 0.681165,
#         'Name': 'Công ty TNHH Shinhan DS Việt Nam',
#         'ApplicationDeadline': '30/11/2024',
#         'LinkCompany': 'https://www.topcv.vn/viec-lam/software-developer-c-c/1505677.html?ta_source=JobSearchList_LinkDetail&u_sr_id=Szls65CkZkgRpYppK3m7LtwVuKQEv2cUdIyD1ljw_1730848215',
#         'Description': "Shinhan DS Vietnam is established to be in charge of the ICT synergy of Shinhan Financial Group in Southeast Asia.\nSDSer's top priority is accelerating the client's business growth by strengthening their ICT systems.\nCollaborate closely with clients and IT professionals in the analysis, development, and testing of banking features & functions (web)\nWrite efficient source code to program complete applications.\nMaintain, modify or create new applications based on requirements.\nIdentify errors, and bugs, and devise solutions to these problems.\nAttend business (banking, stock/ securities & finance), and technical training from Korea's top experts.",
#         'Requirements': 'From 3 years of experience in Software Development/Embedded Development\nProficiency in C/C++ based on Unix, and Linux.\nFamiliar with SQL/Oracle.\nGood to have: ProC and Finance & Banking field.\nGood problem-solving & communication skills.\nDetail-oriented, proactive, and fast learner.',
#         'Benefits': "Our company provides an opportunity to grow as an expert Banking IT in Vietnam\nYou can learn about Banking concepts and know-how from Korea specialized developers\nIn addition, We have plans to expand our company to Indonesia, Myanmar, Philippines\nWe plan to offer training for outstanding employees in Korea. \nLet's Excite you more with our Brilliant perks:\nSalary: You will be happy + Pass-Probation Bonus\nPerformance Bonus: twice a year.\n12 days of annual leaves\nPersonal Allowances for birthdays, marriage, new babies, etc.\nTeam monthly allowance.\nSpecial loan offers and fee waivers from Shinhan Bank.\nCompany Trips, annual/monthly/weekly activities and events.\nHealth check once a year and premium healthcare program. \nEducation Programs and overseas training opportunities.",
#         'Address': '- Hồ Chí Minh: The Mett, Thủ Đức',
#         'WorkingHours': 'Thứ 2 - Thứ 6 (từ 08:00 đến 17:00)',
#         'HowToApply': 'Ứng viên nộp hồ sơ trực tuyến bằng cách bấm Ứng tuyển ngay dưới đây.',
#         'JobId': '881e3129-1059-4c5c-820a-dfc6cc732aae'
#     }
# }
# print(qac_triples[0])

# create answer base
# on the question generated by the context and the context
answer_generation_llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

answer_schema = ResponseSchema(
    name="answer",
    description="an answer to the question"
)

answer_response_schemas = [
    answer_schema,
]

answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)
format_instructions = answer_output_parser.get_format_instructions()

qa_template = """
You are an AI HR Assistant helping employers answer questions about job recruitment. Based on the provided job listings and context, provide clear and professional responses to employee inquiries.

For each question, analyze the context carefully and provide:
1. A direct answer to the question
2. Relevant details from the job listings
3. Additional helpful information if applicable

Please format your response professionally and ensure it aligns with the company's policies and requirements.

Format the output as JSON with the following keys:
answer

question: {question}
context: {context}
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)

messages = prompt_template.format_messages(
    context=qac_triples[0]["context"],
    question=qac_triples[0]["question"],
    format_instructions=format_instructions
)

answer_generation_chain = bare_template | answer_generation_llm

# response = answer_generation_chain.invoke({"content" : messages})
# output_dict = answer_output_parser.parse(response.content)

# for k, v in output_dict.items():
#     print(k)
#     print(v)

# Final dataset combine question, context and answer
for triple in tqdm(qac_triples):
    messages = prompt_template.format_messages(
        context=triple["context"],
        question=triple["question"],
        format_instructions=format_instructions
    )
    response = answer_generation_chain.invoke({"content" : messages})
    try:
        output_dict = answer_output_parser.parse(response.content)
    except Exception as e:
        continue
    triple["answer"] = output_dict["answer"]

# hold the data in a tabular format.
# DataFrame allows for easy data manipulation, analysis, and cleaning, demonstrating the advantages of holding data in a tabular format.
ground_truth_qac_set = pd.DataFrame(qac_triples)


ground_truth_qac_set["context"] = ground_truth_qac_set["context"]
ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer" : "ground_truth"})

# allows the data to be used with Hugging Face's tools and libraries, which are commonly used for machine learning and NLP tasks.
eval_dataset = Dataset.from_pandas(ground_truth_qac_set)

# print(eval_dataset[0])
eval_dataset.to_csv("groundtruth_eval_dataset.csv")

# evaluate_final_result(retrieval_augmented_qa_chain, eval_dataset)
