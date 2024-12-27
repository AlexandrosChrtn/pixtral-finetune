import os
import requests
import praw
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io

"""
Reddit Meme Scraper

This script scrapes memes and their top comments from specified subreddits,
processes the images to a standard format, and saves the data to CSV.

Requirements:
    - Replace "your_client_id" and "your_client_secret" with the ones tied to your account
"""

# Constants
MIN_COMMENT_SCORE = 12
IMAGE_SIZE = 512
IMAGE_QUALITY = 95
MAX_COMMENTS = 10
VALID_IMAGE_EXTENSIONS = ('.jpg', '.png', '.gif', '.jpeg', '.webp')
SUBREDDIT_NAME = "memes"
NUM_POSTS = 1000
OUTPUT_FOLDER = "imgs"
CSV_FILE = "memes_data.csv"

# Configure Reddit API
reddit = praw.Reddit(
    client_id="your_client_id",
    client_secret="your_client_secret",
    user_agent='fetching_reddit_data',
    )

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize data collection
data = []

# Fetch posts
subreddit = reddit.subreddit(SUBREDDIT_NAME)
for category in ['hot', 'top']:
    subreddit_category = (
        subreddit.top(limit=NUM_POSTS, time_filter='year') if category == 'top' 
        else subreddit.hot(limit=NUM_POSTS)
    )
    
    for post in tqdm(subreddit_category, total=NUM_POSTS, desc=f"Fetching {category} posts"):
        
        # Skip if we already processed this post
        if any(item['post_id'] == post.id for item in data):
            continue

        # Filter for image posts
        if not post.url.endswith(VALID_IMAGE_EXTENSIONS):
            continue

        # Fetch top comments
        post.comment_sort = "top"
        post.comments.replace_more(limit=0)
        top_comments = [comment for comment in post.comments[:MAX_COMMENTS]]

        comments_data = []
        for comment in top_comments:
            # Filter out bot comments, comments with brackets like [deleted], and comments with low likes
            if ("am a bot" in comment.body.lower() or 
                "[" in comment.body or 
                "]" in comment.body or 
                comment.score < MIN_COMMENT_SCORE):
                continue
            comments_data.append({
                "comment_body": comment.body,
                "comment_likes": comment.score
            })

        # Download image
        image_path = os.path.join(OUTPUT_FOLDER, f"{post.id}.jpg")
        try:
            img_data = requests.get(post.url, stream=True)
            if img_data.status_code == 200:
                # Load image from bytes
                image = Image.open(io.BytesIO(img_data.content))
                
                # Convert to RGB if necessary (handles PNG with transparency)
                if image.mode in ('RGBA', 'P'):
                    image = image.convert('RGB')
                
                # Resize image to IMAGE_SIZE x IMAGE_SIZE maintaining aspect ratio
                image.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
                
                # Create a new IMAGE_SIZE x IMAGE_SIZE white image
                new_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), 'white')
                
                # Paste the resized image in the center
                offset = ((IMAGE_SIZE - image.size[0]) // 2, (IMAGE_SIZE - image.size[1]) // 2)
                new_image.paste(image, offset)
                
                # Save the final image
                new_image.save(image_path, 'JPEG', quality=IMAGE_QUALITY)
            else:
                print(f"Failed to download image from {post.url} - Status code: {img_data.status_code}")
                continue
        except requests.exceptions.RequestException as e:
            print(f"Request exception while downloading image from {post.url}: {e}")
            continue
        except Exception as e:
            print(f"Error downloading image from {post.url}: {e}")
            continue

        # Add to data
        data.append({
            "post_id": post.id,
            "post_title": post.title,
            "post_url": post.url,
            "image_path": image_path,
            "comments": comments_data
        })

# Save data to CSV
rows = []
for item in data:
    for comment in item["comments"]:
        rows.append({
            "post_id": item["post_id"],
            "post_title": item["post_title"],
            "post_url": item["post_url"],
            "image_path": item["image_path"],
            "comment_body": comment["comment_body"],
            "comment_likes": comment["comment_likes"]
        })

df = pd.DataFrame(rows)
df.to_csv(CSV_FILE, index=False)
print(f"Saved data to {CSV_FILE}. Images saved in {OUTPUT_FOLDER}.")
