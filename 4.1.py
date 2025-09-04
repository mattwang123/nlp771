from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load and split dataset
ds = load_dataset("wics/strategy-qa", revision="refs/convert/parquet")
test_ds = ds["test"]

splits = test_ds.train_test_split(train_size=0.9, seed=42)
train_ds, rest_ds = splits['train'], splits['test']
val_test_splits = rest_ds.train_test_split(train_size=0.5, seed=42)
dev_ds, test_ds_final = val_test_splits['train'], val_test_splits['test']

strategyqa_splits = DatasetDict({
    "train": train_ds,
    "validation": dev_ds,
    "test": test_ds_final,
})

# 2. Tokenizer and preprocessing
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

def preprocess(examples):
    return tokenizer(
        examples["question"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

encoded = strategyqa_splits.map(preprocess, batched=True)
encoded = encoded.rename_column("answer", "label")
encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
train, dev, test = encoded["train"], encoded["validation"], encoded["test"]

# 3. Model (freeze backbone)
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base", num_labels=2
)

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# Then unfreeze only classifier/head parameters
for name, param in model.named_parameters():
    if "classifier" in name or "score" in name:
        param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=10,
    per_device_train_batch_size=64,
    learning_rate=5e-5,
    #weight_decay=0.0001,
    warmup_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to=[],
)

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

# 5. Custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_accuracies = []
        self.dev_accuracies = []
        self.training_finished = False

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Only track during training epochs, not final evaluation
        if (not self.training_finished and 
            metric_key_prefix == "eval" and
            self.state.epoch > 0 and
            self.state.epoch <= self.args.num_train_epochs):
            
            train_metrics = super().evaluate(self.train_dataset, ignore_keys, metric_key_prefix="train")
            self.train_accuracies.append(train_metrics["train_accuracy"])
            self.dev_accuracies.append(metrics["eval_accuracy"])
            
        return metrics

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=dev,
    compute_metrics=compute_metrics,
)

# 6. Train and evaluate
trainer.train()
trainer.training_finished = True

dev_metrics = trainer.evaluate(dev)
test_metrics = trainer.evaluate(test)

print("Final Dev Metrics:", dev_metrics)
print("Final Test Metrics:", test_metrics)

# 7. Plot training vs dev accuracy
epochs = range(1, len(trainer.train_accuracies) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, trainer.train_accuracies, 'r-o', label="Train Accuracy")
plt.plot(epochs, trainer.dev_accuracies, 'b-o', label="Dev (Validation) Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ModernBERT Head-Tuned Classifier: Training Progress")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
print("Plot saved as accuracy_curve.png")