import os
import torch

from datasets import load_dataset
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)
from trl import SFTTrainer, SFTConfig


# # Introduction
# 
# * Datasets:
#     * https://huggingface.co/datasets/tatsu-lab/copa-sent?row=1
# * Models:
#     * openai-community/gpt2
#  
# ***Note:*** *Here we will manually preprocess the input before feeding it to the model. We use `formatting_func` in the SFT API.*

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb
# !pip install -U accelerate peft bitsandbytes transformers trl datasets


import os
import torch
import sys

from datasets import load_dataset
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
    EarlyStoppingCallback,
    set_seed
)
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM,SFTConfig
set_seed(12345)

# import wandb
os.environ['HTTP_PROXY'] = 'http://10.150.1.1:3128'
os.environ['HTTPS_PROXY'] = 'http://10.150.1.1:3128'
os.environ["HF_HUB_OFFLINE"] = "1"

os.environ["WANDB_API_KEY"] = "60316d7e5a0d826b537790b5b8cc04405a1f11ef"
# wandb.init(mode="disabled")

# ## Configuration

num_epochs= 40 
batch_size = 4 #16 #16
num_workers = os.cpu_count()
max_steps = 3000
bf16 = False
fp16 = True
gradient_accumulation_steps = 2
context_length = 1024
logging_steps = 50
save_steps = 50
learning_rate = 2e-5 #0.0001

ft_model= "gpt2-cpt-hr" #sys.argv[1] #"160M"
print("fine tuning model", ft_model)

model_name = '/home/gthakkar/projects/alignment-handbook/data/gpt2-cpt-hr/checkpoint-90000'
out_dir = f'saved_models/copa-sent-completions/{ft_model}'

# ## Load Dataset


dataset = load_dataset("json", data_files={
    "train":"/home/gthakkar/projects/Okapi/datasets/benchich-sent/train_hr.json",
                                           }, num_proc=12)

test_dataset = load_dataset("json", data_files={
    "test":"/home/gthakkar/projects/Okapi/datasets/benchich-sent/test_hr.json"
                                           }, num_proc=12)

print(dataset)


print(dataset['train']['sentence'][0])


full_dataset = dataset['train'].train_test_split(test_size=0.05, shuffle=True,seed=12345)
dataset_train = full_dataset['train']
dataset_valid = full_dataset['test']
 
# print(dataset_train)
# print(dataset_valid)


# for i in range(10):
#     print(dataset_train[i])
#     print('****************')
    
#     text = dataset_train[i]
#     instruction = '### Instruction:\n' + text['instruction']
#     inputs = '\n\n### Input:\n' + text['input']
#     response = '\n\n### Response:\n' + text['output']
    
#     final_text = instruction + inputs + response
#     print(final_text)
#     print('#'*50)


# def preprocess_function(example):
#     """
#     Formatting function returning a list of samples (kind of necessary for SFT API).
#     """
#     text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
#     return text
#  "instruction": "Klasificirajte zadani članak kao pozitivni ili negativni sentiment.",
# Klasificirajte sljedeću rečenicu kao pozitivnu, negativnu ili neutralnu.
    # "input": "Novi automobil je razočaranje. Kočnice su užasne i previše koštaju u odnosu na ponuđene značajke.",
    # "output": "Negativno",

def preprocess_function(example):
    """
    Formatting function returning a list of samples (kind of necessary for SFT API).
    """
    output_texts = []
    for i in range(len(example['sentence'])):
        text = f"### Instruction:\nKlasificirajte sljedeću rečenicu kao pozitivnu, negativnu ili neutralnu.\n\n ### Input:\n{example['sentence'][i]}\n\n ### Response:\n{example['label'][i]}" +tokenizer.eos_token
        output_texts.append(text)
    return output_texts

response_template = " ### Response:"

# ## Model


if bf16:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(dtype=torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name)


print(model)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")


# ## Tokenizer


tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    use_fast=False
)
if "hrtok" in ft_model:
    tokenizer.pad_token = "<|endoftext|>"
else:
    tokenizer.pad_token = tokenizer.eos_token

# print("-1",tokenizer.pad_token)
# if not tokenizer.pad_token:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     print("-2",tokenizer.pad_token)
#     model.resize_token_embeddings(len(tokenizer))

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# ## Training
training_args = SFTConfig(

    output_dir=f"{out_dir}/logs",
    overwrite_output_dir=True,
    eval_strategy='steps',
    weight_decay=0.01,
    load_best_model_at_end=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_strategy='steps',
    save_strategy='steps',
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_total_limit=3,
    bf16=bf16,
    fp16=fp16,
    report_to=['tensorboard'],
    # max_steps=max_steps,
    num_train_epochs=num_epochs,
    dataloader_num_workers=num_workers,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    lr_scheduler_type='constant',
    seed=42,
    max_seq_length=context_length,
    do_eval=True,
    eval_steps=logging_steps,
    run_name=f"{ft_model}-copa-sent"
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=preprocess_function,
    packing=False, #True,  # it was true
    data_collator=collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
)


dataloader = trainer.get_train_dataloader()
for i, sample in enumerate(dataloader):
    print(tokenizer.decode(sample['input_ids'][0]))
    print('#'*50)
    if i % 5:
        break


history = trainer.train(resume_from_checkpoint = False)

print(history)
model.save_pretrained(f"{out_dir}/best_model")
tokenizer.save_pretrained(f"{out_dir}/best_model")

## Inference


from transformers import (
    AutoModelForCausalLM, 
    logging, 
    pipeline,
    AutoTokenizer
)


model = AutoModelForCausalLM.from_pretrained(f'{out_dir}/best_model/')
tokenizer = AutoTokenizer.from_pretrained(f'{out_dir}/best_model/')

tokenizer.pad_token = tokenizer.eos_token


# logging.set_verbosity(logging.CRITICAL)


pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer, max_new_tokens=3, device="cuda")


prompt = """### Instruction:
Klasificirajte sljedeću rečenicu kao pozitivnu, negativnu ili neutralnu.

### Input:

### Response:
"""


print(prompt)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    print ({
    'accuracy': acc,
    'f1': f1,
    'precision': precision,
    'recall': recall})
with open("sent-test-predication.csv","w") as outputfile:
    outputfile.write("label\tprediction\n")
    for i in test_dataset["test"]:

        prompt = f"### Instruction:\nKlasificirajte sljedeću rečenicu kao pozitivnu, negativnu ili neutralnu.\n\n ### Input:\n{i['sentence']}\n\n ### Response:"
        result = pipe(
            prompt
        )
        print(result[0]['generated_text'])
        prediction = result[0]['generated_text'].rsplit("### Response:",1)[-1].strip()
        outputfile.write(i["label"]+"\t"+prediction+"\n")
import pandas as pd
df_results = pd.read_csv("sent-test-predication.csv",sep="\t")

from sklearn.preprocessing import LabelEncoder  

print(df_results["label"].unique(),df_results["prediction"].unique())
le = LabelEncoder()
le.fit_transform(df_results["label"])

print(compute_metrics(le.transform(df_results["prediction"]),le.transform(df_results["label"])))