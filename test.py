from peft import PeftModel, PeftConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

# Load the model and processor
model_id = "mistral-community/pixtral-12b"
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
processor = AutoProcessor.from_pretrained(model_id)

# Load single image using PIL
image_path = "test_img.webp"
image = Image.open(image_path).convert("RGB")
image.thumbnail((512, 512)) # Resize to 512 x 512 as we did on training

# Chat template is applied to the prompt manually
title = "these type of youtubers are the best"
PROMPT = f"<s>[INST]Come up with a comment that will get upvoted by the community for a reddit post in r/memes. Provide the comment body with text and nothing else. The post has title: '{title}' and image:\n[IMG][/INST]"

# First, generate a response with the regular model
inputs = processor(text=PROMPT, images=image, return_tensors="pt").to("cuda")
generate_ids = model.generate(**inputs, max_new_tokens=650, do_sample=True)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# Print the response
print('Pixtral-12B output without fine-tuning:')
print('----'*20)
print(output)
print('----'*20)

# Load the LoRA configuration
peft_config = PeftConfig.from_pretrained("AlexandrosChariton/reddit-pixtral-12B-Lora-v4", use_auth_token=True)
# Apply the LoRA configuration to the base model
lora_model = PeftModel.from_pretrained(model, "AlexandrosChariton/reddit-pixtral-12B-Lora-v4", use_auth_token=True)
inputs = processor(text=PROMPT, images=image, return_tensors="pt").to("cuda")

# Secondly, generate a response with the social media trained model
generate_ids = lora_model.generate(**inputs, max_new_tokens=650)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# Print the response
print('Fine-tuned version of Pixtral-12B:')
print('----'*20)
print(output)
print('----'*20)