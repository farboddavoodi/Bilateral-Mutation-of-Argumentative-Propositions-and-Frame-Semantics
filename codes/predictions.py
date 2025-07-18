import os
import json
import matplotlib.pyplot as plt
from transformers import Trainer

# Function to compute metrics
def compute_metrics_for_predictions(predictions, labels=None):
    """
    Adapts the compute_metrics function for predictions.
    Handles cases where labels might not be available.
    """
    logits = predictions.predictions
    labels = predictions.label_ids if labels is None else labels
    predictions = logits.argmax(axis=-1)

    results = {}

    # 1. Loss and Perplexity (if labels are available)
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

        results["eval_loss"] = sum(per_sample_losses) / len(per_sample_losses)
        results["perplexity"] = math.exp(results["eval_loss"])
        results["loss_per_sample"] = per_sample_losses

    # 2. Token-Level Metrics
    if labels is not None:
        flattened_preds = []
        flattened_labels = []
        for pred, label in zip(predictions, labels):
            for p_token, l_token in zip(pred, label):
                if l_token != -100:  # Exclude padding tokens
                    flattened_preds.append(p_token)
                    flattened_labels.append(l_token)

        if flattened_labels and flattened_preds:
            results["token_accuracy"] = accuracy_score(flattened_labels, flattened_preds)
            results["precision"] = precision_score(flattened_labels, flattened_preds, average="macro")
            results["recall"] = recall_score(flattened_labels, flattened_preds, average="macro")
            results["f1"] = f1_score(flattened_labels, flattened_preds, average="macro")

    # 3. Text-Based Metrics
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

# Function to process predict dataset
def process_predict_dataset(trainer, predict_dataset, output_dir):
    # Generate predictions
    predictions = trainer.predict(predict_dataset)

    # Compute metrics
    metrics = compute_metrics_for_predictions(predictions)

    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, "predict_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Predict dataset metrics saved to {metrics_path}")

    return metrics

# Plot all metrics computed for the predict dataset
def plot_predict_metrics(metrics, title="Predict Dataset Metrics"):
    plt.figure(figsize=(10, 6))
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            plt.bar(key, value)

    plt.title(title)
    plt.ylabel("Metric Value")
    plt.grid(axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

output_dir = "path/to/output_dir"

# Process the predict dataset and compute metrics
predict_metrics = process_predict_dataset(trainer, predict_dataset, output_dir)


plot_predict_metrics(predict_metrics)











from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
                if r_token != -100:  # Exclude padding tokens
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

        results["eval_loss"] = sum(per_sample_losses) / len(per_sample_losses)
        results["perplexity"] = math.exp(results["eval_loss"])
        results["loss_per_sample"] = per_sample_losses

    # 2. Token-Level Metrics
    if labels is not None:
        flattened_preds = []
        flattened_labels = []
        for pred, label in zip(predictions, labels):
            for p_token, l_token in zip(pred, label):
                if l_token != -100:  # Exclude padding tokens
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



from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
import os
import json

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
