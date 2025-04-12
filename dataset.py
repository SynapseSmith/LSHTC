import os
import pandas as pd
from datasets import load_dataset
from constants import *
from utils import idx_to_ltr
from torch.utils.data import Dataset
from dataclasses import dataclass
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from datasets import load_dataset
import time
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder

start_time = time.time()

class RAGPromptGenerator:
    def __init__(self, domain_dict, dataset_path='/home/user06/beaver/LSHTC/data/Data.xlsx', 
                 domain_index_path='/home/user06/beaver/LSHTC/data/DomainVectorDB', 
                 area_index_path='/home/user06/beaver/LSHTC/data/AreaVectorDB', k=5):
        self.dataset = pd.read_excel(dataset_path)
        self.domain_dict = domain_dict
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        self.k = k
        self.domain_index_path = domain_index_path
        self.area_index_path = area_index_path

        self.domain_vectorstore = self._initialize_vectorstore(self.domain_index_path, 'domain')
        self.area_vectorstore = self._initialize_vectorstore(self.area_index_path, 'area')

        self.domain_retriever = self.domain_vectorstore.as_retriever(search_kwargs={"k": k})
        self.area_retriever = self.area_vectorstore.as_retriever(search_kwargs={"k": k})

        # self.llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
        self.prompt_template = hub.pull("lshtc_prompt")
        
        # # 모델 초기화
        # model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

        # # 상위 3개의 문서 선택
        # compressor = CrossEncoderReranker(model=model, top_n=3)

        # # 문서 압축 검색기 초기화
        # self.domain_compression_retriever = ContextualCompressionRetriever(
        #     base_compressor=compressor, base_retriever=self.domain_retriever
        # )
        
        # self.area_compression_retriever = ContextualCompressionRetriever(
        #     base_compressor=compressor, base_retriever=self.area_retriever
        # )

    def _initialize_vectorstore(self, index_path, index_type):
        if os.path.exists(index_path):
            return self._load_vectorstore(index_path)
        else:
            print(f'START CREATING {index_type.upper()} VECTOR STORE.')
            vectorstore = self._create_vectorstore(index_type)
            self._save_vectorstore(vectorstore, index_path)
            return vectorstore

    def _create_vectorstore(self, index_type):
        if index_type == 'domain':
            unique_values = self.dataset[['Domain']].drop_duplicates()
            docs = [Document(page_content=f"Domain: {row['Domain'].strip()}") for _, row in unique_values.iterrows()]
        elif index_type == 'area':
            unique_values = self.dataset[['Domain', 'area']].drop_duplicates()
            docs = [Document(page_content=f"Domain: {row['Domain'].strip()}, Area: {row['area'].strip()}") for _, row in unique_values.iterrows()]
        else:
            raise ValueError("Invalid index type")

        splits = self.text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=self.embeddings)
        return vectorstore

    def _load_vectorstore(self, index_path):
        vectorstore = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"Vectorstore loaded from {index_path}")
        return vectorstore

    def _save_vectorstore(self, vectorstore, index_path):
        if not os.path.exists(index_path):
            os.makedirs(index_path)
        vectorstore.save_local(index_path)
        print(f"Vectorstore saved to {index_path}")

    def generate_rag_prompt(self, abstract, domain=None):
        if domain is None:
            # Domain classification with Plan-and-Solve
            prompt = (
                f"You are a domain expert with extensive knowledge across diverse scientific fields. "
                f"Your task is to classify the following scientific abstract into its most appropriate domain. "
                f"Domains follow a hierarchical structure and may sometimes overlap conceptually. To classify accurately, "
                f"you must first plan your approach and then solve the task step-by-step.\n\n"
                f"Abstract:\n{abstract.strip()}\n\n"
                f"Below is a list of potential domains for this abstract, provided as guidance based on similar content. "
                f"Carefully evaluate the abstract and decide which domain aligns most closely with its main focus:\n"
            )
            retrieved_docs = self.domain_retriever.get_relevant_documents(abstract)
            candidate_domains = [doc.page_content.replace("Domain: ", "") for doc in retrieved_docs]

            for i, domain in enumerate(self.domain_dict.keys()):
                if domain in candidate_domains:
                    prompt += f"{idx_to_ltr(i)}. {domain}\n"

            prompt += (
                "\n**Plan:**\n"
                "1. Identify the key terms, methods, or objectives described in the abstract.\n"
                "2. Compare these elements with the characteristics of the domains provided.\n"
                "3. If the abstract overlaps with multiple domains, prioritize the one that best aligns with the research's core focus.\n"
                "4. If none fit perfectly, provide your best judgment using the given content.\n\n"
                "**Solve:**\n"
                "Based on your analysis and the steps in your plan, determine the most suitable domain. "
                "Respond concisely with a single letter corresponding to the domain, or suggest a new domain if necessary.\n"
            )
        else:
            # Area classification with Plan-and-Solve
            prompt = (
                f"You are a domain expert tasked with identifying the specific subfield (area) within a scientific domain. "
                f"Having already identified the domain, your task is now to classify the following abstract into the most appropriate area within that domain. "
                f"Follow the Plan-and-Solve approach to ensure a structured and accurate classification.\n\n"
                f"Domain: \"{domain}\"\n"
                f"Abstract:\n{abstract.strip()}\n\n"
                f"The following is a list of potential areas within the domain, provided as guidance based on similar abstracts. "
                f"Each area represents a specialized subfield. Carefully evaluate the abstract and determine which area aligns most closely with its content:\n"
            )
            retrieved_docs = self.area_retriever.get_relevant_documents(abstract)
            candidate_areas = [doc.page_content.split(", ")[1].replace("Area: ", "") for doc in retrieved_docs]

            for i, area in enumerate(self.domain_dict[domain]):
                if area in candidate_areas:
                    prompt += f"{idx_to_ltr(i)}. {area}\n"

            prompt += (
                "\n**Plan:**\n"
                "1. Extract the key techniques, objectives, and findings detailed in the abstract.\n"
                "2. Match these elements against the provided areas' descriptions.\n"
                "3. If multiple areas appear relevant, prioritize the one that most closely matches the primary research focus.\n"
                "4. If none fit perfectly, use your expertise to suggest a new area.\n\n"
                "**Solve:**\n"
                "Using your analysis from the plan, identify the most suitable area. "
                "Respond concisely with a single letter corresponding to the area, or suggest a new area if necessary.\n"
            )

        prompt += "Answer:"
        return prompt


@dataclass
class Question:
    abstract: str
    domain: str
    area: str
    domain_dict: dict
    rag_prompt_generator: RAGPromptGenerator
    
    def get_domain_answer_str(self, idx):
        return list(self.domain_dict.keys())[idx]
    
    def get_area_answer_str(self, domain, idx):
        return self.domain_dict[domain][idx]
    
    def _get_prompt(self, include_choices, domain=None):
        if domain is None:
            prompt = f"Predict the domain for the following abstract. Domains are hierarchical.\n\nAbstract:\n{self.abstract.strip()}\n\nDomains:\n"
            if include_choices:
                for i, choice in enumerate(self.domain_dict.keys()):
                    prompt += f"{idx_to_ltr(i)}. {choice}\n"
            prompt += "\nSelect the domain (single letter):\n"
        else:
            prompt = f"Domain: \"{domain}\". Predict the area.\n\nAbstract:\n{self.abstract.strip()}\n\nAreas for \"{domain}\":\n"
            for i, choice in enumerate(self.domain_dict[domain]):
                prompt += f"{idx_to_ltr(i)}. {choice}\n"
            prompt += "\nSelect the area (single letter):\n"
        prompt += "Answer:"
        # print('='*100)
        # print('abstract length:', len(self.abstract))
        # print('prompt length:', len(prompt))
        # print(prompt)
        return prompt
    
    def _get_flat_prompt(self, include_choices, domain=None):
        if domain is None:
            prompt = f"Predict the domain for the following abstract. Domains are hierarchical.\n\nAbstract:\n{self.abstract.strip()}\n\nDomains:\n"
            if include_choices:
                for i, choice in enumerate(self.domain_dict.keys()):
                    prompt += f"{idx_to_ltr(i)}. {choice}\n"
            prompt += "\nSelect the domain (single letter):\n"
        else:
            i = 0
            prompt = f"Domain: \"{domain}\". Predict the area.\n\nAbstract:\n{self.abstract.strip()}\n\nAreas for \"{domain}\":\n"
            for choices in self.domain_dict.values():
                for choice in choices:
                    prompt += f"{idx_to_ltr(i)}. {choice}\n"
                    i += 1
            prompt += "\nSelect the area (single letter):\n"
        prompt += "Answer:"
        print('abstract length:', len(self.abstract))
        print('prompt length:', len(prompt))
        print('='*100)
        return prompt
    
    def _get_poe_prompt(self, mask_dict, domain=None):
        if domain is None:
            prompt = f"You need to solve a task where, given the abstract of a scientific paper, you must sequentially predict the domain and area. The domain and area follow a hierarchical structure, where each domain has specific areas associated with it. Let’s first predict the domain. Here's an abstract:\n\n{self.abstract.strip()}\n\nHere’s a list of possible domains:\n\n"
            for i, choice in enumerate(self.domain_dict.keys()):
                if mask_dict[idx_to_ltr(i)] == -100.0:
                    prompt += f"{idx_to_ltr(i)}. [MASK]\n"
                else:
                    prompt += f"{idx_to_ltr(i)}. {choice}\n"
            prompt += "\nBased on the abstract, identify the most appropriate domain for this paper. Please respond with only single alphabet corresponding to the domain.\n\n"
        else:
            prompt = f"You need to solve a task where, given the abstract of a scientific paper, you must sequentially predict the domain and area. The domain and area follow a hierarchical structure, where each domain has specific areas associated with it. Here's an abstract:\n\n{self.abstract.strip()}\n\nYou have selected appropriate domain as \"{domain}\". Let’s predict the appropriate area. Here are the types of areas possible for the selected domain:\n\n"
            for i, choice in enumerate(self.domain_dict[domain]):
                if mask_dict[idx_to_ltr(i)] == -100.0:
                    prompt += f"{idx_to_ltr(i)}. [MASK]\n"
                else:
                    prompt += f"{idx_to_ltr(i)}. {choice}\n"
            prompt += "\nBased on the abstract and the predicted domain, identify the most appropriate area for this paper. Please respond with only single alphabet corresponding to the area.\n\n"
        return prompt + "Answer:"

    def _get_cot_prompt(self, include_choices, domain=None):
        if domain is None:
            prompt = f"You need to solve a task where, given the abstract of a scientific paper, you must sequentially predict the domain and area. The domain and area follow a hierarchical structure, where each domain has specific areas associated with it. Let’s first predict the domain. Here's an abstract:\n\n{self.abstract.strip()}\n\nHere’s a list of possible domains:\n\n"
            if include_choices:
                for i, choice in enumerate(self.domain_dict.keys()):
                    prompt += f"{idx_to_ltr(i)}. {choice}\n"
            prompt += "\nBased on the abstract, identify the most appropriate domain for this paper. Please respond with only single alphabet corresponding to the domain.\n\n"
        else:
            prompt = f"You need to solve a task where, given the abstract of a scientific paper, you must sequentially predict the domain and area. The domain and area follow a hierarchical structure, where each domain has specific areas associated with it. Here's an abstract:\n\n{self.abstract.strip()}\n\nYou have selected appropriate domain as \"{domain}\". Let’s predict the appropriate area. Here are the types of areas possible for the selected domain:\n\n"
            for i, choice in enumerate(self.domain_dict[domain]):
                prompt += f"{idx_to_ltr(i)}. {choice}\n"
            prompt += "\nBased on the abstract and the predicted domain, identify the most appropriate area for this paper. Please respond with only single alphabet corresponding to the area.\n\n"
        return prompt + "Let's think step by step. "
    
    def _get_ps_prompt(self, include_choices, domain=None):
        if domain is None:
            prompt = f"You need to solve a task where, given the abstract of a scientific paper, you must sequentially predict the domain and area. The domain and area follow a hierarchical structure, where each domain has specific areas associated with it. Let’s first predict the domain. Here's an abstract:\n\n{self.abstract.strip()}\n\nHere’s a list of possible domains:\n\n"
            if include_choices:
                for i, choice in enumerate(self.domain_dict.keys()):
                    prompt += f"{idx_to_ltr(i)}. {choice}\n"
            prompt += "\nBased on the abstract, identify the most appropriate domain for this paper. Please respond with only single alphabet corresponding to the domain.\n\n"
        else:
            prompt = f"You need to solve a task where, given the abstract of a scientific paper, you must sequentially predict the domain and area. The domain and area follow a hierarchical structure, where each domain has specific areas associated with it. Here's an abstract:\n\n{self.abstract.strip()}\n\nYou have selected appropriate domain as \"{domain}\". Let’s predict the appropriate area. Here are the types of areas possible for the selected domain:\n\n"
            for i, choice in enumerate(self.domain_dict[domain]):
                prompt += f"{idx_to_ltr(i)}. {choice}\n"
            prompt += "\nBased on the abstract and the predicted domain, identify the most appropriate area for this paper. Please respond with only single alphabet corresponding to the area.\n\n"
        return prompt + "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step. "
    
    def get_natural_prompt(self, domain=None):
        return self._get_prompt(include_choices=True, domain=domain)
    
    def get_rag_prompt(self, domain=None):
        return self.rag_prompt_generator.generate_rag_prompt(self.abstract, domain)
    
    def get_flat_prompt(self, domain):
        return self._get_flat_prompt(include_choices=True, domain=domain)
    
    def get_brown_prompt(self):
        return self._get_prompt(include_choices=False)
    
    def get_poe_prompt(self, mask_dict, domain=None):
        return self._get_poe_prompt(mask_dict)
    
    def get_cot_prompt(self, domain=None):
        return self._get_cot_prompt(include_choices=True, domain=domain)
    
    def get_ps_prompt(self, domain=None):
        return self._get_ps_prompt(include_choices=True, domain=domain)
    
    def permute_choices(self, perm):
        self.choices = [self.choices[i] for i in perm]
        self.answer_idx = perm.index(self.answer_idx)
        
    def get_labels(self):
        return self.domain, self.area
    
    def get_top_k_domain_area(self):
        self.retrieved_docs = self.rag_prompt_generator.retriever.get_relevant_documents(self.abstract)
        
        candidate_domains = [doc.page_content.split(", ")[0].replace("Domain: ", "") for doc in self.retrieved_docs]
        candidate_areas = [doc.page_content.split(", ")[1].replace("Area: ", "") for doc in self.retrieved_docs]
        
        return candidate_domains, candidate_areas
        
# class QuestionWithExemplar(Question):
#     def __init__(self, parts, choices, answer_idx, exemplar, base_prompt, question_id):
#         super().__init__(question_id, parts, choices, answer_idx)
#         self.exemplar = exemplar
#         self.base_prompt = base_prompt
    
#     def get_natural_prompt(self):
#         prompt = super().get_natural_prompt()
#         if len(self.exemplar):
#             examplar_prompts = [e.get_natural_prompt() for e in self.exemplar]
#             exemplar = "\n\n".join(examplar_prompts)
#             return f"{self.base_prompt}\n\n{exemplar}\n\n{prompt}"
#         else:
#             return f"{self.base_prompt}\n\n{prompt}"
        
#     def get_brown_prompt(self):
#         prompt = super().get_brown_prompt()
#         if len(self.exemplar):
#             exemplar_prompts = [e.get_brown_prompt() for e in self.exemplar]
#             exemplar = "\n\n".join(exemplar_prompts)
#             return f"{self.base_prompt}\n\n{exemplar}\n\n{prompt}"
#         else:
#             return f"{self.base_prompt}\n\n{prompt}"
        
    
# class Exemplar(Question):
#     def get_natural_prompt(self):
#         prompt = super().get_natural_prompt()
#         answer_ltr = idx_to_ltr(self.answer_idx)
#         return f"{prompt} {answer_ltr}"
    
#     def get_brown_prompt(self):
#         prompt = super().get_brown_prompt()
#         return f"{prompt} {self.get_answer_str()}"


class wos_dataset(Dataset):
    def __init__(self, split: int):
        self.split = split
        self.dataset = self.load_wos(self.split)
        self.domain_dict = self.make_domain_dict()
        self.rag_prompt_generator = RAGPromptGenerator(domain_dict=self.domain_dict)
        self.questions = self.make_questions_from_dataset()
        
        
    def __getitem__(self, idx):
        return self.questions[idx]
    
    def __len__(self):
        return len(self.questions)

    def load_wos(self, split: int):
        if split == 5736:
            _split = "WOS5736"
        elif split == 11967:
            _split = "WOS11967"
        elif split == 46985:
            _split = "WOS46985"
        else:
            raise AssertionError("Invalid split")
        
        self.metadata = pd.read_excel(WOS_DIR)
        dataset = load_dataset("HDLTex/web_of_science", _split)['train']
        dataset = dataset.map(self.find_label_text)
        
        return dataset

    def find_label_text(self, examples):
        text = examples['input_data'].strip()
        domain = self.metadata[self.metadata['Abstract'] == text]['Domain'].values[0]
        if len(domain) == 0:
            print(domain)
            raise AssertionError("Domain not found")
        area = self.metadata[self.metadata['Abstract'] == text]['area'].values[0]
        if len(area) == 0:
            print(area)
            raise AssertionError("Area not found")
        
        return {
            "domain": domain.strip(), 
            "area": area.strip()
        }
        
    def make_questions_from_dataset(self):
        questions = []
        for example in self.dataset:
            abstract = example['input_data']
            domain = example['domain']
            area = example['area']
            questions.append(Question(abstract, domain, area, self.domain_dict, self.rag_prompt_generator))
        
        return questions
            
    def get_dataset(self):
        return self.dataset
    
    def make_domain_dict(self):
        domain_dict = {}
        for domain in self.metadata['Domain'].unique().tolist():
            areas = self.metadata[self.metadata['Domain'] == domain]['area'].unique().tolist()
            areas = [area.strip() for area in areas]
            domain_dict[domain.strip()] = areas
            
        return domain_dict
    
if __name__=="__main__":
    dataset = wos_dataset(5736)
    print(dataset[0])
    print(dataset[0].get_natural_prompt())
    print(dataset[0].get_natural_prompt(domain='biochemistry'))