# How to tune an LLM for social media engagement on your own data
Repository with minimal code to fine tune [Pixtral](https://mistral.ai/news/pixtral-12b/) on your own data.

LLMs for the general public like ChatGPT are good at everything. You might ask it to solve a math problem or write the scenario for a TV ad and it will happily obey. The point of this code is to start from a model that is good in everything and make it an expert in one task. To do it you need to provide high quality data for your task and train the model for a few epochs.

For example, if you want your LLM to be able to parse images of your computer screen for example and come up with a text response, you are better off providing several successful examples of what you want to achieve and train the model to perform better for your task.

Here, as an example I provided code to fine tune Pixtral-12B, an LLM that also accepts input signal as input, to produce engaging comments for the popular subsection of Reddit named [r/memes](https://www.reddit.com/r/memes/). This model is based on the original Pixral model from [Mistral](https://mistral.ai/) that was also provided a small but high quality number of comments that were upvoted by the reddit community in the past.

You can use the code here to train a model of your own, for a task you need to solve that involves understanding a visual input. Also, you can test the social media-savvy model that was trained on Reddit comments, using parts of the code in the repo.

As an example, it generated the top comment for the post:

<p align="center">
<img width="546" alt="gh_guide_img" src="https://github.com/user-attachments/assets/5cfbe84c-1eb2-4b05-9923-3f24475baaff" />
</p>

## Table of Contents

- [Installation](#installation)
- [Data Fetching](#data-fetching)
- [Model Training](#model-training)
- [Testing](#testing)
- [Usage](#usage)
- [License](#license)

## Installation
Before moving on, have a quick read on my blogpost about this work, ensure you have enough compute to tackle this. 
To get started, clone the repository and install the required dependencies. Use these commands on your terminal (Ubuntu / MacOS)

```
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

1. Ensure your data is prepared and available in `memes_data.csv` and in the directory `imgs`.
2. If you want to upload it to Hugging Face, add your Hugging Face access token in `login(token="your_hf_token")`. You can generate one [here](https://huggingface.co/settings/tokens).
3. Run the training script:
```
python3 train.py
```

This will fine-tune the model and save it to the specified directory.

- Loads and preprocesses the dataset.
- Applies a chat template to structure the input for the model.
- Fine-tunes the model using LoRA (Low-Rank Adaptation) for efficient training.
- Saves the fine-tuned model and tokenizer.
- Uploads everything to Hugging Face

## Testing

To test the model, whether it is the fine tuned model or the post's version of Pixtral-12B trained on Reddit memes, you need the script `test.py`. If you have a MacBook, replace `"cuda"` with `"mps"`. If you want to try it on a specific post
- Replace `image_path = "test_img.webp"` with the path to the image of the meme, after you download it locally
- Replace `title = ...` with the title of the post
- Run `python3 test.py` and you will see 2 outputs, one of the regular Pixtral-12B and one from the social-media trained Pixtral-12B so you can compare the results

Keep in mind, running this will download Pixtral-12B at first and then the adapter's of the LoRA fine tuned model.

## Usage

If you want the Reddit optimized model, it is already on Hugging Face and you can use it with `test.py` and nothing else. Ensure you have the necessary image and post title, and use the model to generate a comment.

If you want to generalize the code here for a different problem:
- You need to generate your data in a different way. Basically, based on your needs, you can improve the performance of the model on any task that involves a visual signal
- You need to change the part in `train.py` with the prompt inside the function `apply_chat_template` to suit your problem
- You can run the code but you might need to tune a few hyperparameters in case you come up against GPU memory issues or if training is slow, or if the model underperforms in general.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
