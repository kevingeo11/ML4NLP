import pandas as pd
from langchain_huggingface import HuggingFacePipeline
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)


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


model = HuggingFacePipeline.from_model_id(model_id="meta-llama/Llama-3.2-3B-Instruct", task="text-generation", 
    pipeline_kwargs={"max_new_tokens": 256, "return_full_text": False})

print(model.invoke())