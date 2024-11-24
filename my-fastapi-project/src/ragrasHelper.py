import json
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity,
)

from ragas import evaluate
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from basicQAChain import retrieval_augmented_qa_chain
from datasets import Dataset
eval_dataset = Dataset.from_csv("groundtruth_eval_dataset.csv")
def clean_context(context):
    cleaned_list = []
    for item in context:
        # Convert the dictionary to a JSON string
        json_str = json.dumps(item)
        # Remove unwanted characters
        cleaned_str = json_str.replace("{", "").replace("}", "").replace("'", "").replace("[", "").replace("]", "").replace("\n", " ")
        cleaned_list.append(cleaned_str)
    return cleaned_list

def create_ragas_dataset(rag_pipeline, eval_dataset_param):
    rag_dataset = []
    for row in tqdm(eval_dataset_param):
        answer = rag_pipeline.invoke({"question": row["question"]})
        rag_dataset.append(
            {
            "question" : row["question"],
            "answer" : answer["response"].content,
            "contexts": clean_context(answer["context"]),
            "ground_truth" : row["ground_truth"]
            }
        )
    rag_df = pd.DataFrame(rag_dataset)
    rag_eval_dataset = Dataset.from_pandas(rag_df)
    return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset):
    result = evaluate(
        ragas_dataset,
        metrics=[
            # answer_relevancy,
            # faithfulness,
            # context_recall,
            context_precision,
            # answer_correctness,
            # answer_similarity,
        ],
    )
    return result

if __name__ == "__main__":
    data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts': [['"Score": 0.8721453, "Name": "C\\u00f4ng ty TNHH Shinhan DS Vi\\u1ec7t Nam", "ApplicationDeadline": "30/11/2024", "LinkCompany": "https://www.topcv.vn/viec-lam/software-developer-c-c/1505677.html?ta_source=JobSearchList_LinkDetail&u_sr_id=Szls65CkZkgRpYppK3m7LtwVuKQEv2cUdIyD1ljw_1730848215", "Description": "Shinhan DS Vietnam is established to be in charge of the ICT synergy of Shinhan Financial Group in Southeast Asia.\\nSDSers top priority is accelerating the clients business growth by strengthening their ICT systems.\\nCollaborate closely with clients and IT professionals in the analysis, development, and testing of banking features & functions (web)\\nWrite efficient source code to program complete applications.\\nMaintain, modify or create new applications based on requirements.\\nIdentify errors, and bugs, and devise solutions to these problems.\\nAttend business (banking, stock/ securities & finance), and technical training from Koreas top experts.", "Requirements": "From 3 years of experience in Software Development/Embedded Development\\nProficiency in C/C++ based on Unix, and Linux.\\nFamiliar with SQL/Oracle.\\nGood to have: ProC and Finance & Banking field.\\nGood problem-solving & communication skills.\\nDetail-oriented, proactive, and fast learner.", "Benefits": "Our company provides an opportunity to grow as an expert Banking IT in Vietnam\\nYou can learn about Banking concepts and know-how from Korea specialized developers\\nIn addition, We have plans to expand our company to Indonesia, Myanmar, Philippines\\nWe plan to offer training for outstanding employees in Korea. \\nLets Excite you more with our Brilliant perks:\\nSalary: You will be happy + Pass-Probation Bonus\\nPerformance Bonus: twice a year.\\n12 days of annual leaves\\nPersonal Allowances for birthdays, marriage, new babies, etc.\\nTeam monthly allowance.\\nSpecial loan offers and fee waivers from Shinhan Bank.\\nCompany Trips, annual/monthly/weekly activities and events.\\nHealth check once a year and premium healthcare program. \\nEducation Programs and overseas training opportunities.", "Address": "- H\\u1ed3 Ch\\u00ed Minh: The Mett, Th\\u1ee7 \\u0110\\u1ee9c", "WorkingHours": "Th\\u1ee9 2 - Th\\u1ee9 6 (t\\u1eeb 08:00 \\u0111\\u1ebfn 17:00)", "HowToApply": "\\u1ee8ng vi\\u00ean n\\u1ed9p h\\u1ed3 s\\u01a1 tr\\u1ef1c tuy\\u1ebfn b\\u1eb1ng c\\u00e1ch b\\u1ea5m \\u1ee8ng tuy\\u1ec3n ngay d\\u01b0\\u1edbi \\u0111\\u00e2y.", "JobId": "881e3129-1059-4c5c-820a-dfc6cc732aae"', '"Score": 0.8083384, "Name": "C\\u00f4ng ty TNHH VKX", "ApplicationDeadline": "29/11/2024", "LinkCompany": "https://www.topcv.vn/viec-lam/truong-nhom-lap-trinh-developer-leader-yeu-cau-4-nam-kinh-nghiem-thu-nhap-len-toi-45-trieu-thang-lam-viec-tai-ha-noi/1518802.html?ta_source=JobSearchList_LinkDetail&u_sr_id=Szls65CkZkgRpYppK3m7LtwVuKQEv2cUdIyD1ljw_1730848215", "Description": "- Qu\\u1ea3n l\\u00fd \\u0111\\u1ed9i ng\\u0169 ph\\u00e1t tri\\u1ec3n ph\\u1ea7n m\\u1ec1m: L\\u00e3nh \\u0111\\u1ea1o v\\u00e0 h\\u01b0\\u1edbng d\\u1eabn c\\u00e1c l\\u1eadp tr\\u00ecnh vi\\u00ean trong nh\\u00f3m, \\u0111\\u1ea3m b\\u1ea3o ti\\u1ebfn \\u0111\\u1ed9 v\\u00e0 ch\\u1ea5t l\\u01b0\\u1ee3ng c\\u00f4ng vi\\u1ec7c. \\n- Thi\\u1ebft k\\u1ebf v\\u00e0 ph\\u00e1t tri\\u1ec3n ph\\u1ea7n m\\u1ec1m: Tham gia nghi\\u00ean c\\u1ee9u, ph\\u00e1t tri\\u1ec3n, n\\u00e2ng c\\u1ea5p c\\u00e1c s\\u1ea3n ph\\u1ea9m ph\\u1ea7n m\\u1ec1m Geosis, s\\u1ed1 h\\u00f3a d\\u1ecbch v\\u1ee5 c\\u00f4ng, ... v\\u1edbi vai tr\\u00f2 Team Leader\\n- Thi\\u1ebft k\\u1ebf ki\\u1ebfn tr\\u00fac h\\u1ec7 th\\u1ed1ng, Database, ph\\u00e1t tri\\u1ec3n c\\u00e1c ch\\u1ee9c n\\u0103ng \\u0111\\u1ed3ng th\\u1eddi support c\\u00e1c th\\u00e0nh vi\\u00ean trong team\\n- L\\u00e0m c\\u00e1c c\\u00f4ng vi\\u1ec7c kh\\u00e1c theo s\\u1ef1 ph\\u00e2n c\\u00f4ng c\\u1ee7a qu\\u1ea3n l\\u00fd.", "Requirements": "- C\\u00f3 kinh nghi\\u1ec7m l\\u1eadp tr\\u00ecnh Fullstack (.NET-ReactJS) t\\u1eeb 4 n\\u0103m tr\\u1edf l\\u00ean; \\n- Th\\u00e0nh th\\u1ea1o .NET MVC, .Net Core, SQL Server, Postgresql Server, Javascript/Jquery, HTML/CSS; \\n- C\\u00f3 kh\\u1ea3 n\\u0103ng ph\\u00e2n t\\u00edch thi\\u1ebft k\\u1ebf h\\u1ec7 th\\u1ed1ng v\\u00e0 ki\\u1ebfn th\\u1ee9c v\\u1ec1 Design Pattern, OOP...; \\n- C\\u00f3 kinh nghi\\u1ec7m l\\u00e0m vi\\u1ec7c v\\u1edbi MongoDB, Redis Cache, Rabbit MQ, SignalR; \\n- C\\u00f3 ki\\u1ebfn th\\u1ee9c v\\u1ec1 b\\u1ea3o m\\u1eadt v\\u00e0 hi\\u1ec3u bi\\u1ebft v\\u1ec1 Google Adwords API, Cache.", "Benefits": "- M\\u1ee9c l\\u01b0\\u01a1ng: Th\\u1ecfa thu\\u1eadn theo n\\u0103ng l\\u1ef1c, l\\u00ean t\\u1edbi 45.000.000 VN\\u0110/ th\\u00e1ng; \\n- H\\u01b0\\u1edfng \\u0111\\u1ea7y \\u0111\\u1ee7 c\\u00e1c ch\\u1ebf \\u0111\\u1ed9 b\\u1ea3o hi\\u1ec3m theo lu\\u1eadt c\\u1ee7a nh\\u00e0 n\\u01b0\\u1edbc; \\n- Ch\\u1ebf \\u0111\\u1ed9 ngh\\u1ec9 ph\\u00e9p, ngh\\u1ec9 L\\u1ec5, T\\u1ebft theo quy \\u0111\\u1ecbnh c\\u1ee7a Nh\\u00e0 n\\u01b0\\u1edbc; c\\u00e1c ng\\u00e0y ngh\\u1ec9 kh\\u00e1c theo quy \\u0111\\u1ecbnh ri\\u00eang c\\u1ee7a C\\u00f4ng ty (ng\\u00e0y th\\u00e0nh l\\u1eadp C\\u00f4ng ty, ngh\\u1ec9 m\\u00e1t...) \\n- \\u0110\\u01b0\\u1ee3c \\u0111\\u00e0o t\\u1ea1o tr\\u1ef1c ti\\u1ebfp trong qu\\u00e1 tr\\u00ecnh th\\u1ef1c hi\\u1ec7n D\\u1ef1 \\u00e1n, theo l\\u1ed9 tr\\u00ecnh ph\\u00e1t tri\\u1ec3n ri\\u00eang cho t\\u1eebng c\\u00e1 nh\\u00e2n, ph\\u00f9 h\\u1ee3p v\\u1edbi n\\u0103ng l\\u1ef1c;", "Address": "- H\\u00e0 N\\u1ed9i: S\\u1ed1 139 \\u0111\\u01b0\\u1eddng Ng\\u1ecdc H\\u1ed3i, ph\\u01b0\\u1eddng Ho\\u00e0ng Li\\u1ec7t, qu\\u1eadn Ho\\u00e0ng Mai, Ho\\u00e0ng Mai", "WorkingHours": "Th\\u1ee9 2 - Th\\u1ee9 6 (t\\u1eeb 08:00 \\u0111\\u1ebfn 17:00)", "HowToApply": "\\u1ee8ng vi\\u00ean n\\u1ed9p h\\u1ed3 s\\u01a1 tr\\u1ef1c tuy\\u1ebfn b\\u1eb1ng c\\u00e1ch b\\u1ea5m \\u1ee8ng tuy\\u1ec3n ngay d\\u01b0\\u1edbi \\u0111\\u00e2y.", "JobId": "8b0fb8a5-ed6d-4abe-94c6-9deb51328ff0"'],['context 2']],
    'ground_truth': ['The first superbowl was held on January 15, 1967','ground_truth 2']
    }

    data_samples_2 = {
    'question': ['What are the career growth opportunities at Shinhan DS Vietnam?'],
    'answer': ['Career growth opportunities at Shinhan DS Vietnam'],
    'contexts': [
        ['"Score": 0.8721453, "Name": "C\\u00f4ng ty TNHH Shinhan DS Vi\\u1ec7t Nam", "ApplicationDeadline": "30/11/2024", "LinkCompany": "https://www.topcv.vn/viec-lam/software-developer-c-c/1505677.html?ta_source=JobSearchList_LinkDetail&u_sr_id=Szls65CkZkgRpYppK3m7LtwVuKQEv2cUdIyD1ljw_1730848215", "Description": "Shinhan DS Vietnam is established to be in charge of the ICT synergy of Shinhan Financial Group in Southeast Asia.\\nSDSers top priority is accelerating the clients business growth by strengthening their ICT systems.\\nCollaborate closely with clients and IT professionals in the analysis, development, and testing of banking features & functions (web)\\nWrite efficient source code to program complete applications.\\nMaintain, modify or create new applications based on requirements.\\nIdentify errors, and bugs, and devise solutions to these problems.\\nAttend business (banking, stock/ securities & finance), and technical training from Koreas top experts.", "Requirements": "From 3 years of experience in Software Development/Embedded Development\\nProficiency in C/C++ based on Unix, and Linux.\\nFamiliar with SQL/Oracle.\\nGood to have: ProC and Finance & Banking field.\\nGood problem-solving & communication skills.\\nDetail-oriented, proactive, and fast learner.", "Benefits": "Our company provides an opportunity to grow as an expert Banking IT in Vietnam\\nYou can learn about Banking concepts and know-how from Korea specialized developers\\nIn addition, We have plans to expand our company to Indonesia, Myanmar, Philippines\\nWe plan to offer training for outstanding employees in Korea. \\nLets Excite you more with our Brilliant perks:\\nSalary: You will be happy + Pass-Probation Bonus\\nPerformance Bonus: twice a year.\\n12 days of annual leaves\\nPersonal Allowances for birthdays, marriage, new babies, etc.\\nTeam monthly allowance.\\nSpecial loan offers and fee waivers from Shinhan Bank.\\nCompany Trips, annual/monthly/weekly activities and events.\\nHealth check once a year and premium healthcare program. \\nEducation Programs and overseas training opportunities.", "Address": "- H\\u1ed3 Ch\\u00ed Minh: The Mett, Th\\u1ee7 \\u0110\\u1ee9c", "WorkingHours": "Th\\u1ee9 2 - Th\\u1ee9 6 (t\\u1eeb 08:00 \\u0111\\u1ebfn 17:00)", "HowToApply": "\\u1ee8ng vi\\u00ean n\\u1ed9p h\\u1ed3 s\\u01a1 tr\\u1ef1c tuy\\u1ebfn b\\u1eb1ng c\\u00e1ch b\\u1ea5m \\u1ee8ng tuy\\u1ec3n ngay d\\u01b0\\u1edbi \\u0111\\u00e2y.", "JobId": "881e3129-1059-4c5c-820a-dfc6cc732aae"'],
    ],
    'ground_truth': ['Shinhan DS Vietnam offers various career growth opportunities.']
    }
    # dataset = Dataset.from_dict(data_samples_2)
    # print(dataset[0])
    # score = evaluate(dataset,metrics=[
    #         context_precision,
    #     ],)
    # score.to_pandas()
    # print(score)
    rag_data = create_ragas_dataset(retrieval_augmented_qa_chain, eval_dataset)
    print (rag_data[0])
    res = evaluate_ragas_dataset(rag_data)
    print(res)
