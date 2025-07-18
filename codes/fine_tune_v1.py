import transformers
import math
import ast
import numpy as np
import os 
import random
import tokenizers
import webbrowser
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, TextStreamer, TextIteratorStreamer, TrainingArguments
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import pandas as pd 
import datasets
from datasets import load_dataset, Dataset
import requests
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_DATASETS_CACHE"] = "/mnt/c/Users/Rpipc/Desktop/llama/farbod/cache"
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import train_test_split
from evaluate import load
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import psutil 


# functions to use:

class SaveMetricsCallback(TrainerCallback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.metrics_log_path = os.path.join(self.output_dir, "eval_metrics_log.json")
        self.tensorboard_dir = os.path.join(self.output_dir, "tensorboard_logs")
        self.writer = SummaryWriter(self.tensorboard_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Triggered after each evaluation step during training.
        Saves metrics to individual files, appends them to an aggregated log, 
        and logs to TensorBoard.
        """
        if metrics:
            eval_step = state.global_step
            metrics["step"] = eval_step  # Add step information

            # Save metrics for this specific step in a separate JSON file
            step_metrics_path = os.path.join(self.output_dir, f"eval_metrics_step_{eval_step}.json")
            with open(step_metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Saved evaluation metrics for step {eval_step} to {step_metrics_path}")

            # Append metrics to the aggregated log file
            if os.path.exists(self.metrics_log_path):
                with open(self.metrics_log_path, "r") as f:
                    log = json.load(f)
            else:
                log = []

            log.append(metrics)

            with open(self.metrics_log_path, "w") as f:
                json.dump(log, f, indent=4)
            print(f"Appended evaluation metrics for step {eval_step} to {self.metrics_log_path}")

            # Log metrics to the console
            print(f"\n--- Evaluation Metrics at Step {eval_step} ---")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            print(f"---------------------------------------------\n")

            # Log metrics to TensorBoard
            for key, value in metrics.items():
                if key != "step" and isinstance(value, (int, float)):  # Log only numeric values
                    self.writer.add_scalar(key, value, eval_step)
            print(f"Logged metrics for step {eval_step} to TensorBoard.")

    def on_train_end(self, args, state, control, **kwargs):
        """
        Closes the TensorBoard writer when training ends.
        """
        self.writer.close()
        print(f"TensorBoard logs saved to {self.tensorboard_dir}.")

def plot_and_save_metrics(metrics, output_dir, dataset_name="predict"):
    """
    Plots each metric separately and saves them as images in the specified output directory.
    Args:
        metrics (dict): A dictionary of metrics computed for the prediction dataset.
        output_dir (str): Directory to save the images.
        dataset_name (str): Name of the dataset being processed (e.g., "predict").
    """
    plots_dir = os.path.join(output_dir, f"{dataset_name}_metrics_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Iterate over each metric and plot it
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):  
            plt.figure(figsize=(8, 5))
            plt.bar([metric], [value], color='blue')
            plt.title(f"{metric.capitalize()} ({dataset_name.capitalize()} Dataset)")
            plt.ylabel("Value")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Save the plot as an image
            save_path = os.path.join(plots_dir, f"{metric}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot for {metric} to {save_path}")



def load_latest_eval_metrics(output_dir):
    """
    Loads the latest evaluation metrics from the cumulative log file.
    Args:
        output_dir (str): Directory where evaluation metrics are saved.
    Returns:
        dict: Metrics for the most recent evaluation step.
    """
    log_path = os.path.join(output_dir, "eval_metrics_log.json")

    with open(log_path, "r") as f:
        metrics_log = json.load(f)

    # Get the last recorded metrics
    latest_metrics = metrics_log[-1]
    print(f"Loaded metrics for step {latest_metrics['step']}")
    return latest_metrics


def plot_eval_trends(log_path):
    """
    Plots trends for all metrics across evaluation steps.
    Args:
        log_path (str): Path to the cumulative evaluation metrics log file.
    """
    with open(log_path, "r") as f:
        metrics_log = json.load(f)

    steps = [entry["step"] for entry in metrics_log]
    all_metrics = {key: [] for key in metrics_log[0] if key != "step"}

    for entry in metrics_log:
        for key, value in entry.items():
            if key != "step" and isinstance(value, (int, float)):
                all_metrics[key].append(value)

    # Combined plot for all metrics
    plt.figure(figsize=(12, 8))
    for metric, values in all_metrics.items():
        plt.plot(steps, values, label=metric)
    plt.xlabel("Evaluation Step")
    plt.ylabel("Metric Value")
    plt.title("Evaluation Metrics Trends")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Separate plots for each metric
    for metric, values in all_metrics.items():
        plt.figure(figsize=(8, 5))
        plt.plot(steps, values, label=metric, marker='o')
        plt.xlabel("Training Step")
        plt.ylabel(metric)
        plt.title(f"{metric} Over Training Steps")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()


def save_eval_trends(log_path, output_dir):
    """
    Saves individual trend plots for each metric across evaluation steps.
    Args:
        log_path (str): Path to the cumulative evaluation metrics log file.
        output_dir (str): Directory to save the plots.
    """
    plots_dir = os.path.join(output_dir, "eval_trend_plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(log_path, "r") as f:
        metrics_log = json.load(f)

    steps = [entry["step"] for entry in metrics_log]
    all_metrics = {key: [] for key in metrics_log[0] if key != "step"}

    for entry in metrics_log:
        for key, value in entry.items():
            if key != "step" and isinstance(value, (int, float)):
                all_metrics[key].append(value)

    for metric, values in all_metrics.items():
        plt.figure(figsize=(8, 5))
        plt.plot(steps, values, label=metric, marker='o')
        plt.xlabel("Evaluation Step")
        plt.ylabel(metric)
        plt.title(f"{metric.capitalize()} Trend")
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(plots_dir, f"{metric}.png")
        plt.savefig(save_path)
        print(f"Saved trend plot for {metric} to {save_path}")
        plt.close()


def find_and_kill_process_using_port(port):
    """
    Find and kill the process using the specified port.
    """
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port:
                pid = conn.pid
                if pid:  # Ensure PID is valid
                    proc = psutil.Process(pid)
                    print(f"Killing process {proc.name()} (PID: {pid}) using port {port}.")
                    proc.terminate()  # Terminate the process
                    proc.wait()       # Wait for termination
                    return True
    except psutil.AccessDenied:
        print("Access denied to inspect processes. Try running with administrative privileges.")
    except Exception as e:
        print(f"Error while checking port usage: {e}")
    return False

def start_tensorboard(logdir, port):
    """
    Start TensorBoard on the specified port and open it in a browser.
    """
    tensorboard_command = f"tensorboard --logdir={logdir} --port={port}"
    subprocess.Popen(tensorboard_command, shell=True)
    try:
        webbrowser.open(f"http://localhost:{port}")
    except webbrowser.Error:
        print(f"Default browser not found. Please open http://localhost:{port} manually.")


torch.cuda.empty_cache()
seed = 42
# np.random.default_rng = seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


main_results_path = '/mnt/c/Users/Rpipc/Desktop/llama/farbod/f_d/fine_tuned_models/unsloth/SRL_unsloth_Llama-3.2-3B-Instruct-bnb-4bit'
if not os.path.exists(main_results_path):
    os.makedirs(main_results_path, exist_ok=True)

persian_json_data = '/mnt/c/Users/Rpipc/Desktop/llama/farbod/f_d/datas/fa/davodi.csv'
persian_df = pd.read_csv(persian_json_data)
persian_df.drop_duplicates(inplace=True)

prop_roles = set()
for _, row in persian_df.iterrows():
        prop_str = ast.literal_eval(row['propBankTags'])
        prop_roles.update(prop_str)
prop_roles = list(prop_roles)

frame_roles = set()
for _, row in persian_df.iterrows():
    framenet_str = ast.literal_eval(row['frameNetTags'])
    frame_roles.update(framenet_str)
frame_roles = list(frame_roles)

persian_df_tr, persian_df_temp = train_test_split(persian_df, random_state=seed, test_size=.08)
persian_df_val, persian_df_te = train_test_split(persian_df_temp, random_state=seed, test_size=.05)
del persian_df_temp
persian_df_tr.reset_index(drop=True, inplace=True)
persian_df_val.reset_index(drop=True, inplace=True)
persian_df_te.reset_index(drop=True, inplace=True)
persian_df_te.to_csv(os.path.join(main_results_path, "persian_df_test.csv"), index=False)

persian_df_tr = Dataset.from_pandas(persian_df_tr)
persian_df_val = Dataset.from_pandas(persian_df_val)
persian_df_te = Dataset.from_pandas(persian_df_te)

max_seq_length = 8192 
dtype = None 
load_in_4bit = True # 
model_path = '/mnt/c/Users/Rpipc/Desktop/llama/Llama-3.2-3B-Instruct-bnb-4bit'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

frame_prompt_template = """<|start_header_id|>system<|end_header_id|>
Please generate FrameNet roles for the provided text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
You are an expert in the field of Semantic Role Labeling and lexical resources especially FrameNet.
You know anything about how to label sentence tokens with FrameNet frame elements.
Please use the following text:"+ {}+"Here are the frame elements you have to use for labeling:" + {}+"
Your task is to generate frame elements for the provided text.
The output should be a list of frame elements in a list format. IF the token do not have any frame element, put 'O'.
Make sure that you do NOT use any elements other than the ones I provided in this prompt.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{}
"""

prop_prompt_template = """<|start_header_id|>system<|end_header_id|>
Please generate PropBank roles for the provided text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
You are an expert in the field of Semantic Role Labeling and lexical resources especially PropBank.
You know anything about how to label sentence tokens with PropBank roles.
Please use the following text:"+ {}+"Here are the propbank roles you have to use for labeling:" + {}+"
Your task is to generate PropBank roles for the provided text.
The output should be a list of roles in a list format. IF a token does not have any role, put 'O'.
Make sure that you do NOT use any roles other than the ones I provided in this prompt.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{}
"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_function(examples):
    words = examples["words"]
    frams = examples["frameNetTags"]
    props = examples["propBankTags"]
    texts = []

    # Generate FrameNet prompts
    for word, fram in zip(words, frams):
        text = frame_prompt_template.format(word, str(frame_roles), fram) + EOS_TOKEN
        
        texts.append(text)

    # Generate PropBank prompts
    for word, prop in zip(words, props):
        text = prop_prompt_template.format(word, str(prop_roles), prop) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

main_dataset = persian_df_tr.map(formatting_prompts_function, remove_columns=persian_df_tr.column_names, batched=True,)
eval_dataset = persian_df_val.map(formatting_prompts_function, remove_columns=persian_df_val.column_names, batched=True,) 
test_dataset = persian_df_te.map(formatting_prompts_function, remove_columns=persian_df_te.column_names, batched=True,) 

main_dataset = main_dataset.shuffle(seed=seed)
eval_dataset = eval_dataset.shuffle(seed=seed)
test_dataset = test_dataset.shuffle(seed=seed)


#
# main_dataset = main_dataset.select(range(128))
# eval_dataset = eval_dataset.select(range(30))
# test_dataset = test_dataset.select(range(2))
#
output_dir = os.path.join(main_results_path, "checkpoints")

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


port = 6006
if find_and_kill_process_using_port(port):
    print(f"Port {port} was in use and the process has been terminated.")
else:
    print(f"Port {port} is free.")

# Start TensorBoard
start_tensorboard(main_results_path, port)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=main_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size= 2,
        gradient_accumulation_steps=16,
        eval_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="steps",
        save_steps=500,#500
        logging_strategy="steps",
        logging_steps=500,#500
        eval_strategy="steps",
        eval_steps=500, #500
        output_dir=output_dir,
    ),

    callbacks=[SaveMetricsCallback(output_dir)]  
)

trainer.train()

save_pth = os.path.join(main_results_path, "lora")
if not os.path.exists(save_pth):
    os.makedirs(save_pth, exist_ok=True)
model.save_pretrained(save_pth) 
tokenizer.save_pretrained(save_pth)

latest_eval_metrics = load_latest_eval_metrics(output_dir)
plot_and_save_metrics(latest_eval_metrics, output_dir, dataset_name="eval")

log_path = os.path.join(output_dir, "eval_metrics_log.json")
plot_eval_trends(log_path)
save_eval_trends(log_path, output_dir)
