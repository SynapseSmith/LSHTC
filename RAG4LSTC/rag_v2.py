import os
import getpass
import bs4
import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

def main():
    load_dotenv()

    # 단계 1: 문서 로드(Load Documents)

    loader1 = CSVLoader(file_path='../data/odp_categories.csv')
    docs1 = loader1.load()

    print(f'The number of categories: {len(docs1)}')
    print(docs1[0])

    loader2 = CSVLoader(file_path='../data/odp_top_categories.csv')
    docs2 = loader2.load()

    print(f'The number of top-level categories: {len(docs2)}')
    print(docs2[0])

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    splits1 = text_splitter.split_documents(docs1)
    splits2 = text_splitter.split_documents(docs2)

    # 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
    # 벡터스토어를 생성합니다.
    vectorstore1 = FAISS.from_documents(documents=splits1, embedding=OpenAIEmbeddings())
    vectorstore2 = FAISS.from_documents(documents=splits2, embedding=OpenAIEmbeddings())

    # 단계 4: 검색(Search)
    retriever1 = vectorstore1.as_retriever(search_kwargs={"k": 5})
    retriever2 = vectorstore2.as_retriever(search_kwargs={"k": 3})

    # 단계 5: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = hub.pull("lshtc_prompt2")

    # """
    # You are an assistant for hierarchical text classification tasks. Use the following candidate_categories to classify the given text into its appropriate category. You must choose only one category from the candidate categories.
    # Text: {text}
    # Candidate Categories: {candidate_categories}
    # Classification:
    # """

    # 단계 6: 언어모델 생성(Create LLM)
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

    def format_docs(docs):
        # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
        return "\n\n".join(doc.page_content for doc in docs)

    # 단계 7: 체인 생성(Create Chain)
    rag_chain = (
            {"top_level_categories": retriever2 | format_docs, "candidate_categories": retriever1 | format_docs, "text": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    name_odpid_dict = {}
    with open("../data/CID_Print.txt", "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            id_odpid_name = line.strip().split("\t")
            name_odpid_dict[id_odpid_name[2].replace(",", "")] = id_odpid_name[1]

    # 단계 8: 체인 실행(Run Chain)
    # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
    text_x= []
    gt_y = []

    with open("../data/odp.wsdm.test.all", "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            id_name = line.strip().split(" ", maxsplit=1)
            text_x.append(id_name[1])
            gt_y.append((id_name[0]))

    print(len(text_x))
    print(len(gt_y))

    start_time = time.time()
    positive_count = 0
    name_list = name_odpid_dict.keys()
    count = 0
    with open("../data/inference.txt", "w", encoding='utf-8') as f:
        for idx, text in enumerate(text_x):
            if count % 100 == 0 and count != 0:
                print('count: ', count)
                print(positive_count/count)
            count += 1
            response = rag_chain.invoke(text)
            if not response.startswith("T"):
                response_list = response.split(" ")
                if response.strip('C') and len(response_list) <= 3:
                    response = response_list[-1]
                elif '"' in response:
                    response_list = response.split('\"')
                    for candidate in response_list:
                        if candidate.startswith("Top"):
                            response = candidate
                            continue
            f.write(response + '\n')
            if response not in name_list:
                continue
            if name_odpid_dict[response] == gt_y[idx]:
                positive_count += 1

    print('Micro-F1 Score: ', positive_count/len(gt_y))
    inference_time = (time.time() - start_time) / 60
    print('Inference time: ', inference_time)


if __name__ == "__main__":
    main()