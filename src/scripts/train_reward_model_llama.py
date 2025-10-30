import torch, json, os, wandb
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from transformers import LlamaForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import EvalPrediction
from datasets import load_dataset, Dataset
from datasets import concatenate_datasets
from collections import Counter
# from datasets import load_metric
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

first_call_preprocess_function = True

# 1. Load and process the local dataset
def load_and_process_dataset(file_path, model_name, max_length=512, type='pro', version='version_b'):  # 'pro' or 'con'; 'version_a' or 'version_b'
    assert type == 'pro' or type == 'con', "Type must be 'pro' or 'con'"
    
    # Load dataset
    dataset = load_dataset('json', data_files=file_path)
    
    # Filter dataset based on 'impact_stance'
    dataset = dataset.filter(lambda example: example['impact_stance'] == type)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define preprocessing function
    def preprocess_function(examples):
        # Concatenate list of paths into a single string
        relation_ship = 'supporting' if type == 'pro' else 'attacking'
        
        if version == 'version_a':
            '''Version A implementation'''
            texts = [f"You are given a chain of arguments, each one supporting or attacking the previous one. The first argument is: {path[-1]} The second last one is: {path[1]} The last one is: {path[0]} Now you need to determine the impact of the last one to the second last one, given their relationship {relation_ship}. Output only a number among 0, 1, or 2 in your response. 0 means not impactful; 1 means medium impactful; 2 means impactful." for path in examples['path']]
        elif version == 'version_b':
            '''Version B implementation'''
            texts = []
            for path, path_labels in zip(examples['path'], examples['path_labels']):
                # context = ' '.join(f"The {i}th claim is: {item} " for i, item in enumerate(list(reversed(path))))
                context = ''
                for i, item in enumerate(list(reversed(path))):
                    context += f"The {i+1}th claim is: {item} "
                    if i > 0 and path_labels[-i] == 'pro':
                        context += "This claim is supporting the previous claim.\n"
                    elif i > 0 and path_labels[-i] == 'con':
                        context += "This claim is supporting the previous claim.\n"
                    else:
                        context += "\n"
                texts.append(f"You are given a chain of arguments, each one supporting or attacking the previous one. {context}. Now you need to determine the impact of the last claim to the second last one. Output only a number among 0, 1, or 2 in your response. 0 means not impactful; 1 means medium impactful; 2 means impactful.")
        else:
            assert False, "Invalid version"

        global first_call_preprocess_function
        if first_call_preprocess_function:
            print('[prompt]', texts[0])
            first_call_preprocess_function = False
        
        # Tokenize texts
        tokenized = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
        
        # Add labels
        label_map = {"NOT IMPACTFUL": 0, "MEDIUM IMPACT": 1, "IMPACTFUL": 2}
        tokenized['labels'] = [label_map[label] for label in examples['impact_label']]
        
        return tokenized
    
    # Apply preprocessing
    processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

    # Implement resampling
    def resample_dataset(dataset):
        # Convert labels to tensor and count samples for each class
        labels_tensor = torch.tensor(dataset['labels'])
        label_counts = labels_tensor.bincount()
        max_count = label_counts.max().item()
        
        resampled_datasets = []
        for label in range(len(label_counts)):
            class_dataset = dataset.filter(lambda example: example['labels'] == label)
            
            # Calculate how many times to repeat the class dataset
            repeat_factor = min(max_count // len(class_dataset), 3) #NOTE: <=3 times for each sample
            remainder = max_count % len(class_dataset)
            
            # Manually repeat the dataset
            repeated_datasets = [class_dataset] * repeat_factor
            resampled_class = concatenate_datasets(repeated_datasets)
            
            # Add additional samples if needed
            # if remainder > 0:
            #     additional_samples = class_dataset.shuffle().select(range(remainder))
            #     resampled_class = concatenate_datasets([resampled_class, additional_samples])
            
            resampled_datasets.append(resampled_class)

            print('[resample_dataset]', label, len(class_dataset), len(resampled_class))
        
        # Combine all resampled datasets
        return concatenate_datasets(resampled_datasets).shuffle(seed=42)

    # Apply resampling to the train split
    if 'train' in file_path:
        processed_dataset['train'] = resample_dataset(processed_dataset['train'])

    # Set format for PyTorch
    processed_dataset.set_format('torch')
    
    return processed_dataset

# 2. Load LLaMA model and Tokenizer with corrected settings
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model with the correct number of labels
    model = LlamaForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )  # Adjusted to 3
    
    # Set the pad_token_id in the model's configuration
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

# 2.5 metrics
def compute_metrics(p: EvalPrediction):
    predictions = p.predictions
    label_ids = p.label_ids
    
    # If predictions are logits (usually a 2D array), we need to take argmax
    if len(predictions.shape) == 2:
        preds = np.argmax(predictions, axis=-1)
    else:
        preds = predictions  # If already prediction categories, use directly
        
    print('[compute_metrics]', preds, label_ids)
    
    # Calculate accuracy
    accuracy = accuracy_score(label_ids, preds)
    
    # Calculate multi-class F1 score
    f1 = f1_score(label_ids, preds, average='macro')
    
    # Calculate Kappa score
    kappa = cohen_kappa_score(label_ids, preds)
    
    # Combine all metrics
    results = {
        'accuracy': accuracy,
        'f1': f1,
        'kappa': kappa
    }
    
    return results

def compute_confusion_matrix(p: EvalPrediction):
    '''reference vs. prediction'''
    predictions = p.predictions
    label_ids = p.label_ids
    
    if len(predictions.shape) == 2:
        preds = np.argmax(predictions, axis=-1)
    else:
        preds = predictions  
        
    print('[compute_metrics]', preds, label_ids)

    cnt = np.zeros((3, 3))
    for pred, label in zip(preds, label_ids):
        cnt[label][pred] += 1
    
    return cnt

# 3. Fine-tune the model
def fine_tune_model(model, tokenizer, train_dataset, val_dataset, run_name):
    # Set training arguments
    os.environ["WANDB_PROJECT"] = "debating"  # name your W&B project
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./results", run_name + '_' + current_time)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,  
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_only_model=True, 
        per_device_train_batch_size=8,  # Adjust based on GPU memory
        per_device_eval_batch_size=8,
        num_train_epochs=3,  # Adjust as needed
        learning_rate=1e-5,
        logging_dir='./logs',
        logging_steps=100,
        run_name=run_name,
        fp16=False,
        bf16=True,
        tf32=False,
    )

    model.use_weighted_loss = False #NOTE: try the regression idea
    if model.use_weighted_loss:
        print('[NOTE] Using weighted loss')

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    trainer.train()

    predictions = trainer.predict(val_dataset)
    metrics = compute_metrics(predictions)
    metrics = {f'final/{k}': v for k, v in metrics.items()}
    print(metrics)
    wandb.log(metrics)

# 4. Main function
def main():
    # Load and process the dataset
    # type = 'pro'
    type = 'con'
    version = 'version_a'
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    train_dataset = load_and_process_dataset('dataset/kialo/kialo/kialo_path.train.jsonl', model_name, type=type, version=version)['train']
    val_dataset = load_and_process_dataset('dataset/kialo/kialo/kialo_path.valid.jsonl', model_name, type=type, version=version)['train']

    print(train_dataset)
    print(val_dataset)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Fine-tune the model
    run_name = f"llama_{type}_{version}_resampling"
    fine_tune_model(model, tokenizer, train_dataset, val_dataset, run_name)


class LLM:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=-1).item()

def evaluate_support_strength(model, motion, argument1, argument2, history):
    relation_ship = 'supporting'
    prompt = f"You are given a chain of arguments, each one supporting or attacking the previous one. The first argument is: {history[0]} The second last one is: {history[-3]} The last one is: {history[-1]} Now you need to determine the impact of the last one to the second last one, given their relationship {relation_ship}. Output only a number among 0, 1, or 2 in your response. 0 means not impactful; 1 means medium impactful; 2 means impactful."
    return model(prompt)

def evaluate_defense_strength(model, motion, argument1, argument2, history):
    relation_ship = 'attacking'
    prompt = f"You are given a chain of arguments, each one supporting or attacking the previous one. The first argument is: {history[0]} The second last one is: {history[-2]} The last one is: {history[-1]} Now you need to determine the impact of the last one to the second last one, given their relationship {relation_ship}. Output only a number among 0, 1, or 2 in your response. 0 means not impactful; 1 means medium impactful; 2 means impactful."
    return model(prompt)

if __name__ == "__main__":
    main()