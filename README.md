# language_extrapolation
This is an official repository for the paper, [How language models extrapolate outside the training data: A case study in Textualized Gridworld](https://arxiv.org/abs/2406.15275).
## 1. Env. Setup
```
conda create -n [your virtual env. name] python=3.9
conda activate [your virtual env. name]
bash setup.sh
```
___
## 2. Data Preparation
### 2-1. Make skeleton data
```
# Training
python -m envs.gridworld.data.make_jsonl --num 50000 --split train --max_world_x_size 10 --max_world_y_size 10

# Inference
python -m envs.gridworld.data.make_jsonl --num 3000 --split inference --max_world_x_size 20 --max_world_y_size 20
```
### 2-2. Make text data using skeleton data

Our data generation code has 3 main parameters:

`-- first_only`: whether to experiment optimal planning analysis (model trains to generate the whole sequence of actions without actual interaction) or reachable planning analysis (model trains to generate action by action, with corresponding textual observation after each action).

True: Optimal analysis

False: Reachable analysis

___
`-- cot_type`: type of cot, each meaning as follows:

none: Model dosen't vebalize anything (None).

backtrack: Model trains to verbalize a backtrack trace of the target sequence first (CoT).

all_possible_backtrack: Cognitive map (Marking Deadend)

all: Variant of the cogntive map, exclusion of marking deadend and backtracking (w.o. Marking Backtrack)

all_backtrack: Variant of the cogntive map, exclusion of marking deadend (w.o. Marking)

possible: Variant of the cogntive map, exclusion of all sampling and backtracking (w.o. All Backtrack)

possible_backtrack: Variant of the cogntive map, exclusion of all sampling  (w.o. All)

all_possible: Variant of the cogntive map, exclusion of backtracking (w.o. Backtrack)


___
`-- when_thought`:

first-step: Forward construction of the map (FWD)

first-step-reversed: Backward construction of the map (BWD)
___
See Section 5.2 for detailed meaning of each experiments.

In this README file, we provide data generation examples for 6 main cases in Table 1 of our paper:


Case 1: Optimal analysis, None
```
python data/make_sft_gridworld_from_data.py --method vanilla --basic_mode 5 --cot_type none --thought look-ahead --first_only True --when_thought first-step
```

Case 2: Optimal analysis, CoT
```
python data/make_sft_gridworld_from_data.py --method vanilla --basic_mode 5 --cot_type backtrack --thought look-ahead --first_only True --when_thought first-step
```

Case 3: Optimal analysis, Cognitive map(BWD w.o. MARKING) **(Best performance in Optimal analysis)**

```
python data/make_sft_gridworld_from_data.py --method vanilla --basic_mode 5 --cot_type all_backtrack --thought look-ahead --first_only True --when_thought first-step-reversed
```

Case 4: Reachable analysis, None
```
python data/make_sft_gridworld_from_data.py --method vanilla --basic_mode 5 --cot_type none --thought look-ahead --first_only True --when_thought first-step
```

Case 5: Reachable analysis, CoT
```
python data/make_sft_gridworld_from_data.py --method vanilla --basic_mode 5 --cot_type backtrack --thought look-ahead --first_only True --when_thought first-step
```

Case 6: Reachable analysis, Cognitive map(BWD MARKING DEADEND) **(Best performance in Reachable analysis)**
```
python data/make_sft_gridworld_from_data.py --method vanilla --basic_mode 5 --cot_type all_possible_backtrack --thought look-ahead --first_only True --when_thought first-step-reversed
```



## 3. Training code
```bash
IFS=',' read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
node_num=${#gpus[@]}
num_workers=${#gpus[@]}

model_name="meta-llama/Meta-Llama-3-8B"
sft_data_path="path_to_your_data"
save_dir="output_directory"
sft_model_name="name_of_your_trained_model"

batch_size=16
lr=2e-5
micro_batch_size=2
epoch=1
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20000 fastchat/train/train.py \
    --model_name_or_path ${model_name} \
    --data_path ${sft_data_path} \
    --bf16 True \
    --output_dir ${save_dir}${sft_model_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 125 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess False
```

## 4. Testing code

### 4-1. single GPU
Case 1: Optimal planning analysis
```bash
CUDA_VISIBLE_DEVICES=0 python -m eval_agent.main_inference_vllm_single --agent_config fastchat --exp_config gridworld --split test_3000 --model_path ${save_dir}${cur_model_name} --max_tokens_to_generate 8192 --basic_mode 5 --part_num 1 --part_idx 0 --output_dir ${test_output_dir}
```
Case 2: Reachable planning analysis
```bash
CUDA_VISIBLE_DEVICES=0 python -m eval_agent.main_inference_vllm --agent_config fastchat --exp_config gridworld --split test_3000 --model_path ${save_dir}${cur_model_name} --max_tokens_to_generate 8192 --basic_mode 5 --part_num 1 --part_idx 0 --output_dir ${test_output_dir}
```
### 4-2. multi GPU
Case 1: Optimal planning analysis
```bash
IFS=',' read -ra gpus <<< "0,1,2,3,4,5,6,7"
node_num=${#gpus[@]}
num_workers=${#gpus[@]}
worker_idx=0
for ((j=0;j<${num_workers};j=j+1)); do
    echo "Launch the model on gpu ${j}"
    CUDA_VISIBLE_DEVICES=${gpus[$((${worker_idx} % ${node_num}))]} python -m eval_agent.main_inference_vllm_single --agent_config fastchat --exp_config gridworld --split test_3000 --model_path ${save_dir}${cur_model_name} --max_tokens_to_generate 8192 --basic_mode 5 --part_num ${num_workers} --part_idx ${j} --output_dir ${test_output_dir}&
    worker_idx=$(($worker_idx+1))
done
```
Case 2: Reachable planning analysis
```bash
IFS=',' read -ra gpus <<< "0,1,2,3,4,5,6,7"
node_num=${#gpus[@]}
num_workers=${#gpus[@]}
worker_idx=0
for ((j=0;j<${num_workers};j=j+1)); do
    echo "Launch the model on gpu ${j}"
    CUDA_VISIBLE_DEVICES=${gpus[$((${worker_idx} % ${node_num}))]} python -m eval_agent.main_inference_vllm_single --agent_config fastchat --exp_config gridworld --split test_3000 --model_path ${save_dir}${cur_model_name} --max_tokens_to_generate 8192 --basic_mode 5 --part_num ${num_workers} --part_idx ${j} --output_dir ${test_output_dir}&
    worker_idx=$(($worker_idx+1))
done
```
## 5. visualization code
```
python -m eval_agent.visualize_success --exp_config gridworld --split test_3000 --model_path ${save_dir}${cur_model_name} --output_dir ${test_output_dir} --xsize 10 --ysize 10 --test_ysize 20
```

## Appendix. Few-shot experiment
### Appendix A. LLaMA-3 8B
```bash
IFS=',' read -ra gpus <<< "0,1,2,3"
node_num=${#gpus[@]}
num_workers=${#gpus[@]}
worker_idx=0
for ((j=0;j<${num_workers};j=j+1)); do
    echo "Launch the model on gpu ${j}"
    CUDA_VISIBLE_DEVICES=${gpus[$((${worker_idx} % ${node_num}))]} python -m eval_agent.main_inference_vllm_single --agent_config fastchat --exp_config gridworld --split test_3000 --model_path meta-llama/Meta-Llama-3-8B --basic_mode 5 --part_num ${num_workers} --part_idx ${j} --output_dir ${test_output_dir} --n_icl ${n_shot}&
    worker_idx=$(($worker_idx+1))
done
```
### Appendix B. LLaMA-3 70B
```bash
IFS=',' read -ra gpus <<< "0,1,2,3"
node_num=${#gpus[@]}
num_workers=1
node_per_worker=$((${node_num} / ${num_workers}))

worker_idx=0
for ((j=0;j<${num_workers};j=j+1)); do
    echo "gpu : ${gpus[@]:$((${node_per_worker} * j)):${node_per_worker}}"
    gpu_idx=(${gpus[@]:$((${node_per_worker} * j)):${node_per_worker}})
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gpu_idx[*]}") python -m eval_agent.main_inference_vllm_single --agent_config fastchat --exp_config gridworld --split test_3000 --model_path meta-llama/Meta-Llama-3-70B --basic_mode 5 --part_num ${num_workers} --part_idx ${j} --output_dir ${test_output_dir} --n_icl ${n_shot} --num_gpus ${node_per_worker}&
done
```
### Appendix C. OpenAI API (GPT-4o, GPT-o1)
```bash
num_workers=
for ((j=0;j<${num_workers};j=j+1)); do
    python -m eval_agent.main_inference_gpt_single --agent_config fastchat --exp_config gridworld --split test_3000 --model_path ${model_name} --basic_mode 5 --part_num ${num_workers} --part_idx ${j} --output_dir ${test_output_dir} --n_icl ${n_shot}&
done
```
