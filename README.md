# pixtral-finetune
Repository with minimal code to fine tune Pixtral on your own data.

## Table of Contents

- [Installation](#installation)
- [Data Fetching](#data-fetching)
- [Model Training](#model-training)
- [Testing](#testing)
- [Usage](#usage)
- [License](#license)

## Installation

To get started, clone the repository and install the required dependencies:

```
bash
git clone https://github.com/AlexandrosChrtn/pixtral-finetune.git
cd pixtral-finetune
pip install -r requirements.txt
```

## Data Fetching

The `fetch_data.py` script is used to scrape memes and their top comments from Reddit. It processes the images to a standard format and saves the data to a CSV file.

### Usage

1. Configure your Reddit API credentials in the script:
   - Replace `"your_client_id"` and `"your_client_secret"` with your Reddit API credentials found in https://www.reddit.com/prefs/apps after logging in your Reddit account.

2. Run the script to fetch data:
```
python3 fetch_data.py
```

## Model Training

The `train.py` script is used to fine-tune the Pixtral model using the data fetched by `fetch_data.py`.

### Usage

1. Ensure your data is prepared and available in `memes_data.csv`.

2. Run the training script:
```
python train.py
```

This will fine-tune the model and save it to the specified directory.

### Key Features

- Loads and preprocesses the dataset.
- Applies a chat template to structure the input for the model.
- Fine-tunes the model using LoRA (Low-Rank Adaptation) for efficient training.
- Saves the fine-tuned model and tokenizer.
- Uploads everything to Hugging Face, if you add your Hugging Face access token in `login(token="your_hf_token")`. You can generate one in https://huggingface.co/settings/tokens

## Testing

To test the model, wether it is the fine tuned model or my version of Pixtral-12B trained on Reddit memes, you need the script `test.py`. If you want to try it on a specific post
- Replace `image_path = "test_img.webp"` with the path to the image of the meme, after you download it locally
- Replace `title = ...` with the title of the post
- Run `python3 test.py` and you will see 2 outputs, one of the regular Pixtral-12B and one from the social-media trained Pixtral-12B so you can compare the results

Keep in mind, running this will download Pixtral-12B at first and then the adaptor's of the LoRA fine tuned model I generated

## Usage

After training, you can use the fine-tuned model to generate comments for new Reddit meme posts. Ensure you have the necessary image and post title, and use the model to generate a comment. If you want to generalize the code here for a different problem:
- You need to generate your data in a different way. Basically, based on your needs, you can improve the performance of the model on any task that involves a visual signal
- You need to change the part in `train.py` with the prompt inside the function `apply_chat_template` to suit your problem
- You can run the code but you might need to tune a few hyperparameters in case you come up against GPU memory issues or if training is slow, or if the model underperforms in general.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.