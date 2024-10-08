
# Cyberbullying Detection API

The **Cyberbullying Detection API** is designed to fetch comments from public Instagram posts or reels and analyze them for potential hate speech. The API identifies and flags harmful comments across various categories such as:

- **Age**
- **Religion**
- **Sexism**
- **Racism**

This solution provides an efficient way to monitor and mitigate harmful online interactions by using cutting-edge machine learning techniques.


## Disclaimer
This API fetches Instagram comments from **public posts and reels only**. It does not scrape or access private posts, private accounts, or any restricted content. Please be mindful of Instagram's Terms of Use and Data Policy.

## Model Overview

The core of this API is a powerful machine learning model based on **DistilBERT**, a distilled version of BERT (Bidirectional Encoder Representations from Transformers). DistilBERT retains the same language understanding capabilities as BERT but is much more efficient and faster.

The model has been fine-tuned on a diverse dataset of comments to detect different forms of hate speech, achieving a **remarkable accuracy of 95%** in identifying and categorizing hate speech.

### Categories Detected:
- **Age-related comments**
- **Religious insults**
- **Sexist remarks**
- **Racist statements**

## How It Works

1. **Input**: The API accepts a link to a public Instagram post or reel.
2. **Processing**: The API fetches the comments from the provided post.
3. **Analysis**: Each comment is analyzed and categorized into one of the above hate speech categories.
4. **Output**: The results are returned, highlighting any flagged comments along with their respective hate speech category.

## Features
- **Real-time comment fetching**: Automatically retrieves comments from any public post or reel.
- **High accuracy**: The model has been optimized for detecting hate speech with 95% accuracy.
- **Fast and efficient**: Powered by DistilBERT, the API is both lightweight and fast.
- **Multi-category detection**: Capable of flagging hate speech related to age, religion, sexism, and racism.

## Usage

### Prerequisites
- You need an Instagram account to fetch comments.

### API Endpoints

- **POST** `api/v1/analyze`
  - Input: JSON containing a link to a public Instagram post or reel.
  - Output: JSON response containing flagged comments and their categories.

Example Request:
```json
{
  "link": "https://www.instagram.com/p/ABC12345/"
}
```

Example Response:
```json
{
  "predictions": [
    {
      "username":"user1",
      "text": "You're too old for this!",
      "sentiment": "Age"
    },
    {
      "username":"user2",
      "text": "This religion is a joke!",
      "sentiment": "Religion"
    }
  ]
}
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Cyberbullying-detection.git
   ```
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   python fast.py
   ```

## Future Work

- Expanding the model to cover more categories of hate speech.
- Implementing real-time monitoring for live comment sections.
- Adding support for other social media platforms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or raise an issue.

## About Me

Hi! I'm the developer behind the Cyberbullying Detection API. I'm passionate about using technology, AI, and machine learning to solve real-world problems. Currently, I'm a final-year Computer Science and Engineering student with experience in projects ranging from real-time object detection to sentiment analysis and heart arrhythmia classification. I'm always open to collaborating on innovative projects, especially in the areas of AI and social good.

You can check out some of my other projects on my [GitHub profile](https://github.com/kanhaiyachatla) or connect with me on [LinkedIn](https://www.linkedin.com/in/kanhaiya-chatla-b208a6176?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3B6NKl4LJSSu6vGZ873P9oxA%3D%3D).

---

Feel free to customize any details according to your exact project implementation.