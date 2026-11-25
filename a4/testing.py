import pandas as pd
from datasets import DatasetDict, load_dataset
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

## Step 1
tmp_data = pd.read_json("ori_pqal.json").T
# some labels have been defined as "maybe", only keep the yes/no answers
tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

documents = pd.DataFrame({"abstract": tmp_data.apply(lambda row: (" ").join(row.CONTEXTS+[row.LONG_ANSWER]), axis=1),
             "year": tmp_data.YEAR})
questions = pd.DataFrame({"question": tmp_data.QUESTION,
             "year": tmp_data.YEAR,
             "gold_label": tmp_data.final_decision,
             "gold_context": tmp_data.LONG_ANSWER,
             "gold_document_id": documents.index})


## Step 3.2: Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

metadatas = [{"id": idx} for idx in documents.index]
texts = text_splitter.create_documents(texts=documents.abstract.tolist(), metadatas=metadatas)

print(texts[0].page_content)


print(texts[42].page_content)