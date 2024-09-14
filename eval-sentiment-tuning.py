
from transformers import (
    AutoModelForCausalLM, 
    logging, 
    pipeline,
    AutoTokenizer
)

from datasets import load_dataset

ft_model= "gpt2-cpt-hr" #sys.argv[1] #"160M"
print("fine tuning model", ft_model)

model_name = '/home/gthakkar/projects/alignment-handbook/data/gpt2-cpt-hr/checkpoint-90000'
out_dir = f'saved_models/copa-sent-completions/{ft_model}'

test_dataset = load_dataset("json", data_files={
    "test":"/home/gthakkar/projects/Okapi/datasets/benchich-sent/test_hr.json"
                                           }, num_proc=12)


model = AutoModelForCausalLM.from_pretrained(f'{out_dir}/best_model/')
tokenizer = AutoTokenizer.from_pretrained(f'{out_dir}/best_model/')

tokenizer.pad_token = tokenizer.eos_token


# logging.set_verbosity(logging.CRITICAL)


pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer,max_new_tokens=3, device="cuda")


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
preds = []
labels = []
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
