import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
import numpy as np
from datasets import DatasetDict
from huggingface_hub import login, upload_folder, create_repo
from transformers import Trainer, TrainingArguments
import torch

# Load the model and processor
model_id = "mistral-community/pixtral-12b"
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

# Ensure the processor has padding and a chat template for structured prompts
processor.tokenizer.pad_token = "<pad>"
processor.tokenizer.pad_token_id = 11
processor.tokenizer.chat_template = processor.chat_template

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=92,  # Rank of the LoRA matrix
    lora_alpha=12, # Scaling factor for LoRA
    lora_dropout=0.15,
    use_dora=False,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'language_model.lm_head', 'up_proj', 'down_proj', 'gate_proj',
                   "multi_modal_projector.linear_1", "multi_modal_projector.linear_2"]
)

# Apply the LoRA configuration to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load the dataset from a CSV file
dataset = load_dataset("csv", data_files="memes_data.csv")

# Convert the dataset to a pandas DataFrame for easier manipulation
df = dataset["train"].to_pandas()

# Extract unique post IDs for splitting the dataset
unique_post_ids = df['post_id'].unique()

# Randomly split post IDs into training and testing sets
np.random.seed(42)  # Ensure reproducibility
test_post_ids = np.random.choice(unique_post_ids, size=int(0.01 * len(unique_post_ids)), replace=False)
train_post_ids = np.setdiff1d(unique_post_ids, test_post_ids)

# Split the DataFrame into training and testing sets based on post IDs
train_df = df[df['post_id'].isin(train_post_ids)]
test_df = df[df['post_id'].isin(test_post_ids)]

# Convert the DataFrames back to Hugging Face datasets
train_dataset = dataset["train"].from_pandas(train_df)
test_dataset = dataset["train"].from_pandas(test_df)

# Print the sizes of the train and test datasets
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Define a function to apply the chat template to each example
def apply_chat_template(example):
    try:
        # Skip examples with long comments or URLs
        if len(example["comment_body"]) > 240 or "https://" in example["comment_body"]:
            return {"prompt": "", "image": None}
        
        # Load the image from the specified path
        imgpath = example["image_path"]  # Assuming your dataset has image paths
        image = Image.open(imgpath).convert("RGB")

        # Construct the chat messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "content": "Come up with a comment that will get upvoted by the community for a reddit post in r/memes. Provide the comment body with text and nothing else. The post has title: '" + example["post_title"] + "' and image:\n"},
                    {"type": "image"}  # Represents the image input for the model
                ]
            },
            {
                "role": "assistant",
                "content": example["comment_body"]  # Expected output
            },
        ]
        
        # Apply the chat template using the processor
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"prompt": prompt, "image": image}
    except Exception as e:
        print(f"Error processing example {example}: {e}")
        return None  # Skip this example if an error occurs

# Apply the chat template function to the datasets
new_train_dataset = train_dataset.map(apply_chat_template, batched=False)
new_test_dataset = test_dataset.map(apply_chat_template, batched=False)

# Combine the processed datasets into a DatasetDict
new_dataset = DatasetDict({
    'train': new_train_dataset,
    'test': new_test_dataset
})

# Filter out examples where the mapping function returned None
new_dataset = new_dataset.filter(lambda x: len(x['prompt']) > 1)

# Define a function to tokenize the data
def tokenize_function(example):
    # Tokenize the text and image inputs
    inputs = processor(
        text=example["prompt"],
        images=example["image"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=1350
    )
    # Adjust the tensor dimensions
    inputs['pixel_values'] = inputs['pixel_values'][0][0]
    inputs['input_ids'] = inputs['input_ids'].squeeze(0)
    inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
    
    # Prepare labels for training
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore padding in loss calculation
    inputs["labels"] = labels    
    
    return inputs

# Apply tokenization to the dataset
tokenized_dataset = new_dataset.map(tokenize_function)

# Clear CUDA cache to free up memory
torch.cuda.empty_cache()

# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=250,
    logging_steps=150,
    gradient_accumulation_steps=3,
    save_steps=400,
    per_device_train_batch_size=1,  # Adjust based on hardware capabilities
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    fp16=True,  # Use mixed precision if supported
    report_to="none",
    log_level="info",
    learning_rate=2e-5,
    max_grad_norm=2,
    lr_scheduler_type='linear'
)

# Initialize the Trainer for model fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=processor.tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./my_llm_dir")
processor.tokenizer.save_pretrained("./my_llm_dir")

print('Upload to Hugging Face!')
# Login to Hugging Face (you'll need your token from https://huggingface.co/settings/tokens)
login(token="your_hf_token")

# Replace with your repo name on Hugging Face
repo_id = "your_hf_name/my_awesome_LLM"

try:
    # Attempt to create the repo if it doesn't exist
    create_repo(repo_id.split('/')[-1], private=True, use_auth_token=True)  # Set private=True to make the repo private
    print(f"Repo {repo_id.split('/')[-1]} created and set to private.")
except Exception as e:
    print(f"Repo {repo_id.split('/')[-1]} already exists or could not be created: {e}")

# Upload the saved model to the Hugging Face repository
upload_folder(
    folder_path="my_llm_dir",
    repo_id=repo_id,
    use_auth_token=True,
    path_in_repo="."  # This pushes the contents directly to the root of the repo
)