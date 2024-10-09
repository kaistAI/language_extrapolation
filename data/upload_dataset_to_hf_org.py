from dataclasses import dataclass, field
from typing import Optional
import transformers
from datasets import load_dataset, Dataset, DatasetDict
import os
import sys
sys.path.append(os.path.abspath("./"))

from fastchat.train.train import make_supervised_data_module

def main(data_args, model_args, training_args, saving_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_path=model_args.model_name_or_path)
    
    def gen(dataset):
        for idx in range(len(dataset)):
            yield dataset[idx]
            
    train_dataset = Dataset.from_generator(gen, gen_kwargs={"dataset": data_module['train_dataset']})
    if data_module["eval_dataset"]:
        eval_dataset = Dataset.from_generator(gen, gen_kwargs={"dataset": data_module['eval_dataset']})
        dataset_dict = DatasetDict({"train_dataset": train_dataset, "eval_dataset": eval_dataset})
    else:
        dataset_dict = DatasetDict({"train_dataset": train_dataset})
    
    file_name = os.path.basename(data_args.data_path)
    dataset_name = os.path.splitext(file_name)[0]
    if saving_args.token == None:
        dataset_dict.push_to_hub(f"{saving_args.org}/{model_args.model_name_or_path.split('/')[-1]}-{dataset_name}", private=True)
    else:
        dataset_dict.push_to_hub(f"{saving_args.org}/{model_args.model_name_or_path.split('/')[-1]}-{dataset_name}", token=saving_args.token, private=True)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    report_to: str = field(default="wandb")

@dataclass
class SavingArguments:
    org: str = field(
        metadata={
            "help": "the name of Huggingface organization which you want to upload dataset"
        }
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Huggingface token which have WRITE authentication"
        }
    )

if __name__=="__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, SavingArguments)
    )
    model_args, data_args, training_args, saving_args = parser.parse_args_into_dataclasses()
    main(data_args, model_args, training_args, saving_args)