from tqdm import tqdm
import pandas as pd

from ragrasHelper import create_ragas_dataset, evaluate_ragas_dataset

def evaluate_final_result(retrieval_augmented_qa_chain, eval_dataset):
    basic_qa_ragas_dataset =  create_ragas_dataset(retrieval_augmented_qa_chain, eval_dataset)
    print(basic_qa_ragas_dataset[0])
    basic_qa_ragas_dataset.to_csv("basic_qa_ragas_dataset.csv")
    basic_qa_result = evaluate_ragas_dataset(basic_qa_ragas_dataset)
    print(basic_qa_result)
