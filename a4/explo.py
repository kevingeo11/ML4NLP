import pandas as pd
from datasets import DatasetDict, load_dataset
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

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


print(questions.iloc[0].question)
print(documents.iloc[0].abstract)

## Step 2 
model = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    model_kwargs={"torch_dtype": "auto"},
    pipeline_kwargs={"max_new_tokens": 256, "return_full_text": False}
)

print(model.invoke("Can you explain to me what a mitochondria is?")) # TODO Include prompt and output in output file


## Step 3.1: Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

query_embedding = embeddings.embed_query("This is a test document.")

print(len(query_embedding)) # prints embedding_dim: 768


## Step 3.2: Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    add_start_index=True,

)

metadatas = [{"id": idx} for idx in documents.index]
texts = text_splitter.create_documents(texts=documents.abstract.tolist(), metadatas=metadatas)

print(texts[0].page_content) # prints the first chunk

## Step 3.3: Vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
)
vector_store.add_documents(texts[:10]) # TODO change this!

results = vector_store.similarity_search_with_score(
    "What is programmed cell death?", k=3
)
for res, score in results:
    print(f"* [DIST={score:3f}] {res.page_content} [{res.metadata}]")

## Step 4 
## Option B: Build a RAG chain based on LangChain Open Tutorial

retriever = vector_store.as_retriever() 

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    prompt
    | model
    | StrOutputParser()
)
runnable_parallel_object = RunnableParallel(context=retriever, question=RunnablePassthrough())
rag_chain = runnable_parallel_object.assign(answer=chain)

idx = 0
q = questions.iloc[idx].question
out = rag_chain.invoke(q)

print("QUESTION:", out["question"])
print("RETRIEVED:", [d.metadata["id"] for d in out["context"]],
      "| GOLD:", questions.iloc[idx].gold_document_id)
print("CONTEXT:", out["context"][0].page_content[:500])
print("ANSWER:", out["answer"])

match_groundtruth = ["yes" if answer["context"][metadata["id"] == documents.iloc[idx]]
#print("Groundtruth context: ",


answers = []
retrieved_document= []
for i in range(3): 

    q = questions.iloc[i]
    out = rag_chain.invoke(q.question)
    answers.append(out["answer"])
    retrieved_document.append(q.gold_document_id in [d.metadata["id"] for d in out["context"]])


def check(answer, ground_truth):

    if 

    if answer == ground_truth:
        return True
    else:
        return False



for idx, answer in enumerate(answers):
    
    answer = answer.strip()
    answer = answer.lower()
    
    groundtruth = question.gold_id[idx].lower()

    if answer not in ["yes", "no"]:
        answer = "wrong format"
    
    check
    
