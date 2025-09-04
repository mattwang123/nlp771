from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn as nn
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

# 3. Model with LoRA
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base", num_labels=2
)

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=1, alpha=1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features
            
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scaling = alpha / rank
        
    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output

# Freeze entire model
for param in model.parameters():
    param.requires_grad = False

# Apply LoRA to first suitable layer
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and 'classifier' not in name:
        lora_params = module.in_features + module.out_features
        if abs(lora_params - 1538) < 200:
            parent_names = name.split('.')[:-1]
            child_name = name.split('.')[-1]
            
            parent_module = model
            for parent_name in parent_names:
                parent_module = getattr(parent_module, parent_name)
            
            lora_layer = LoRALayer(module, rank=1, alpha=1)
            setattr(parent_module, child_name, lora_layer)
            break

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="results_lora",
    num_train_epochs=10,
    per_device_train_batch_size=64,
    learning_rate=5e-3,
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
        self.training_finished = False  # Flag to stop tracking

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
trainer.training_finished = True  # Stop tracking after training

dev_metrics = trainer.evaluate(dev)
test_metrics = trainer.evaluate(test)

print("Final Dev Metrics:", dev_metrics)
print("Final Test Metrics:", test_metrics)

# 7. Plot results
epochs = range(1, len(trainer.train_accuracies) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, trainer.train_accuracies, 'r-o', label="Train Accuracy")
plt.plot(epochs, trainer.dev_accuracies, 'b-o', label="Dev (Validation) Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ModernBERT LoRA Classifier: Training Progress")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lora_accuracy_curve.png", dpi=300)
print("Plot saved as lora_accuracy_curve.png")