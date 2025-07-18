import transformers
import math
import ast
import numpy as np
import os 
import random
import tokenizers
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


# functions to use:

def split_chat_template(example):
    if "<|start_header_id|>assistant<|end_header_id|>\n" in example["text"]:
        parts = example["text"].split("<|start_header_id|>assistant<|end_header_id|>\n")
        return {"input": parts[0].strip(), "label": parts[1].split('\n<|eot_id|>')[0].strip()}
    else:
        return {"input": example["text"], "label": ""}  
        
def tokenize_chat_data(example):
    inputs = tokenizer(
        example["input"], 
        padding="longest", 
        truncation=True, 
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["label"], 
            padding="longest",
            truncation=True, 
        )

    inputs["labels"] = labels["input_ids"]
    return inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    results = {}

    # Token-Level Accuracy Function (Custom)
    def compute_token_accuracy(predicted_ids, label_ids):
        total_tokens = 0
        correct_tokens = 0

        for pred, ref in zip(predicted_ids, label_ids):
            for p_token, r_token in zip(pred, ref):
                if r_token != -100:  
                    total_tokens += 1
                    if p_token == r_token:
                        correct_tokens += 1

        return (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0

    # 1. Loss and Perplexity
    if labels is not None:
        logits_tensor = torch.tensor(logits)
        labels_tensor = torch.tensor(labels)

        per_sample_losses = []
        for sample_logits, sample_labels in zip(logits_tensor, labels_tensor):
            loss = F.cross_entropy(
                sample_logits.view(-1, sample_logits.size(-1)),
                sample_labels.view(-1),
                reduction="none"
            )
            per_sample_loss = loss.view(sample_labels.size(0), -1).mean(dim=1).mean().item()
            per_sample_losses.append(per_sample_loss)

        results["manual_eval_loss"] = sum(per_sample_losses) / len(per_sample_losses)
        results["perplexity"] = math.exp(results["manual_eval_loss"])
        results["loss_per_sample"] = per_sample_losses

    # 2. Token-Level Metrics
    if labels is not None:
        flattened_preds = []
        flattened_labels = []
        for pred, label in zip(predictions, labels):
            for p_token, l_token in zip(pred, label):
                if l_token != -100:  
                    flattened_preds.append(p_token)
                    flattened_labels.append(l_token)

        if flattened_labels and flattened_preds:
            # Accuracy using sklearn's accuracy_score
            results["accuracy_score"] = accuracy_score(flattened_labels, flattened_preds)

            # Accuracy using the custom function
            results["token_accuracy"] = compute_token_accuracy(predictions, labels)

            # Other token-level metrics
            results["precision"] = precision_score(flattened_labels, flattened_preds, average="macro")
            results["recall"] = recall_score(flattened_labels, flattened_preds, average="macro")
            results["f1"] = f1_score(flattened_labels, flattened_preds, average="macro")

    # 3. Decoded Text-Based Metrics
    if hasattr(tokenizer, "batch_decode"):
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        if labels is not None:
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            def compute_exact_match(predictions, references):
                matches = [
                    pred.strip() == ref.strip()
                    for pred, ref in zip(predictions, references)
                ]
                return sum(matches) / len(matches) * 100

            results["exact_match"] = compute_exact_match(decoded_predictions, decoded_labels)

            # BLEU
            results["bleu"] = bleu.compute(
                predictions=[pred.strip() for pred in decoded_predictions],
                references=[[ref.strip()] for ref in decoded_labels]
            )["bleu"]

            # ROUGE
            results["rougeL"] = rouge.compute(
                predictions=[pred.strip() for pred in decoded_predictions],
                references=[ref.strip() for ref in decoded_labels]
            )["rougeL"].mid.fmeasure

            # METEOR
            results["meteor"] = meteor.compute(
                predictions=[pred.strip() for pred in decoded_predictions],
                references=[[ref.strip()] for ref in decoded_labels]
            )["meteor"]

    return results
    
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


seed = 42
# np.random.default_rng = seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")

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
print(persian_df_tr.shape)
print(persian_df_val.shape)
print(persian_df_te.shape)
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
main_dataset = main_dataset.select(range(2))
eval_dataset = eval_dataset.select(range(2))
test_dataset = test_dataset.select(range(2))
#
processed_eval_dataset = eval_dataset.map(split_chat_template)
tokenized_eval_dataset = processed_eval_dataset.map(
    tokenize_chat_data, 
    batched=True, 
    remove_columns=['input', 'label']
)

# Preprocess predict dataset
processed_predict_dataset = test_dataset.map(split_chat_template, remove_columns=['text'])
tokenized_predict_dataset = processed_predict_dataset.map(
    tokenize_chat_data, 
    batched=True, 
    remove_columns=['input', 'label']
)


output_dir = os.path.join(main_results_path, "checkpoints")

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = main_dataset,
#     eval_dataset = val_dataset,
#     dataset_text_field = "text",
#     max_seq_length = max_seq_length,
#     dataset_num_proc = 2,
#     packing = False,
#     args = TrainingArguments(
#         # eval_steps= 1,
#         eval_strategy='epoch',
#         per_device_train_batch_size = 4,
#         per_device_eval_batch_size= 4,
#         gradient_accumulation_steps = 4,
#         # Use num_train_epochs = 1, warmup_ratio for full training runs!
#         warmup_steps = 5,
#         num_train_epochs = 4,
#         # max_steps = 30000, 
#         learning_rate = 2e-4,
#         fp16 = not is_bfloat16_supported(),
#         bf16 = is_bfloat16_supported(),
#         # logging_steps = 1,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         # save_strategy="steps", 
#         save_strategy="epoch", 
#         logging_strategy = "epoch", 
#         output_dir = output_dir,
#         # save_steps=132,
#     ),
# )

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=main_dataset,
    # eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
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
        save_steps=1,#500
        logging_strategy="steps",
        logging_steps=1,#500
        # eval_strategy="steps",
        # eval_steps=1, #500
        output_dir=output_dir,
        # do_eval=False
    ),
    compute_metrics=compute_metrics,  
    callbacks=[SaveMetricsCallback(output_dir)]  
)


trainer.train()

save_pth = os.path.join(main_results_path, "lora")
if not os.path.exists(save_pth):
    os.makedirs(save_pth, exist_ok=True)
model.save_pretrained(save_pth) 
tokenizer.save_pretrained(save_pth)



predictions = trainer.predict(tokenized_predict_dataset)
decoded_inputs = tokenizer.batch_decode(tokenized_predict_dataset["input_ids"], skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(tokenized_predict_dataset["labels"], skip_special_tokens=True)
decoded_predictions = tokenizer.batch_decode(predictions.predictions.argmax(axis=-1), skip_special_tokens=True)


losses = predictions.metrics["loss_per_sample"]

results_path = os.path.join(main_results_path, "test_predictions_with_metrics.csv")

pred_df = pd.DataFrame({
    "input": decoded_inputs,
    "label": decoded_labels,
    "prediction": decoded_predictions,
    "loss": losses,  
})
pred_df.to_csv(results_path, index=False)

print(f"Results saved to {results_path}")

# Load and plot the latest evaluation metrics
# a. Plot Metrics for Latest Evaluation Step
latest_eval_metrics = load_latest_eval_metrics(main_results_path)
plot_and_save_metrics(latest_eval_metrics, main_results_path, dataset_name="eval")

# b. Plot Trends Across All Evaluation Steps
log_path = os.path.join(main_results_path, "eval_metrics_log.json")
plot_eval_trends(log_path)
save_eval_trends(log_path, main_results_path)


# Generate predictions and metrics for the predict dataset
predictions = trainer.predict(test_dataset)
metrics = predictions.metrics

# c. Plot Metrics for Prediction Dataset
# Save and plot metrics for the predict dataset
metrics_path = os.path.join(main_results_path, "predict_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {metrics_path}")
plot_and_save_metrics(metrics, main_results_path, dataset_name="predict")













# pred_df.to_csv(os.path.join(os.path.dirname(persian_json_data), "test_predictions.csv"), index=False)

# try:
#     # Merge to 16bit
#     model.save_pretrained_merged("/home/data/Tavakoli/unpacker/decoder_based/fine tuned models/unsloth/saved/for vllm/16_bit/Unpacker-on16kb-Qwen2-1.5B-Instruct-bnb-merge16bit", tokenizer, save_method = "merged_16bit",)
# except:
#     pass


# try:
#     model.save_pretrained_merged("/home/data/Tavakoli/unpacker/decoder_based/fine tuned models/unsloth/saved/for vllm/lora_adapter/Unpacker-on16kb-Qwen2-1.5B-Instruct-bnb-merge4bit", tokenizer, save_method = "lora",)
# except:
#     pass

# try:
#     # Save to 8bit Q8_0
#     model.save_pretrained_gguf("/home/data/Tavakoli/unpacker/decoder_based/fine tuned models/unsloth/saved/ggufs/8bit Q8_0/Unpacker-on16kb-Qwen2-1.5B-Instruct-bnb-16bit-GGUF-8bitQ8_0", tokenizer,)
# except:
#     pass
    
# try:
#     # Save to 16bit GGUF
#     model.save_pretrained_gguf("/home/data/Tavakoli/unpacker/decoder_based/fine tuned models/unsloth/saved/ggufs/16bit/Unpacker-on16kb-Qwen2-1.5B-Instruct-bnb-16bit-GGUF-16bit", tokenizer, quantization_method = "f16")
# except:
#     pass
    
# try:
#     # Save to q4_k_m GGUF
#     model.save_pretrained_gguf("/home/data/Tavakoli/unpacker/decoder_based/fine tuned models/unsloth/saved/ggufs/q4_k_m/Unpacker-on16kb-Qwen2-1.5B-Instruct-bnb-16bit-GGUF-q4_k_m", tokenizer, quantization_method = "q4_k_m")
# except:
#     pass

# try:
#     # Merge to 4bit
#     model.save_pretrained_merged("/home/data/Tavakoli/unpacker/decoder_based/fine tuned models/unsloth/saved/for vllm/4_bit/Unpacker-on16kb-Qwen2-1.5B-Instruct-bnb-4bit", tokenizer, save_method = "merged_4bit_forced",)
# except:
#     pass