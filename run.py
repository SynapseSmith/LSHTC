import torch
import time
import os
import pandas as pd

from argparse import ArgumentParser
from tqdm import tqdm

from constants import *
from dataset import wos_dataset
from model import LlamaModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--quantize", type=bool, default=True)
    parser.add_argument("--dataset", type=str, default="wos")
    parser.add_argument("--task", type=str, default='rag') #"naiive")
    parser.add_argument("--log_dir", type=str, default="./LSHTC/logs")
    parser.add_argument("--save_chunk_size", type=int, default=100)
    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.task, time.strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Load model
    model = LlamaModel(args.model_name, args.quantize)

    # Load dataset
    if args.dataset == "rcv1":
        pass
    elif args.dataset == "wos":
        dataset = wos_dataset(5736)
    
    with open(os.path.join(log_dir, "log.csv"), "w") as f:
        f.write("idx,abstract,true_domain,predicted_domain,true_area,predicted_area\n")
    
    # Predict
    question_list, response_list = [], []
    for i, question in enumerate(tqdm(dataset)):
        response = model.process_question_rag(question)
        question_list.append(question)
        response_list.append(response)
        
        if i % args.save_chunk_size == 0:
            print("Saving chunk")
            with open(os.path.join(log_dir, "log.csv"), "a") as f:
                for i in tqdm(range(len(question_list))):
                    question = question_list[i]
                    response = response_list[i]
                    question.abstract = question.abstract.replace("\n", "...").replace(",", ".")
                    f.write(f"{i},{question.abstract},{response.true_domain},{response.predicted_domain},{response.true_area},{response.predicted_area}\n")
            question_list, response_list = [], []
        
    print("Saving last chunk")
    with open(os.path.join(log_dir, "log.csv"), "a") as f:
        for i in tqdm(range(len(question_list))):
            question = question_list[i]
            response = response_list[i]
            question.abstract = question.abstract.replace("\n", "...").replace(",", ".")
            f.write(f"{i},{question.abstract},{response.true_domain},{response.predicted_domain},{response.true_area},{response.predicted_area}\n")
    question_list, response_list = [], []
        
if __name__ == "__main__":
    main()