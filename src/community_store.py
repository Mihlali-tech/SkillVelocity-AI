import json
import os
from datetime import datetime

DATA_FILE = "data/community_posts.json"

def _ensure_storage():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump([], f)

def get_all_posts():
    _ensure_storage()
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def add_post(author, title, content, tag):
    _ensure_storage()
    posts = get_all_posts()
    new_post = {
        "id": len(posts) + 1,
        "author": author or "Anonymous Learner",
        "title": title,
        "content": content,
        "tag": tag,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "replies": []
    }
    posts.insert(0, new_post) # Latest first
    with open(DATA_FILE, "w") as f:
        json.dump(posts, f, indent=4)
    return True

def add_reply(post_id, author, reply_content):
    _ensure_storage()
    posts = get_all_posts()
    for post in posts:
        if post["id"] == post_id:
            post["replies"].append({
                "author": author or "Community Member",
                "content": reply_content,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            break
    with open(DATA_FILE, "w") as f:
        json.dump(posts, f, indent=4)
    return True