import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Loading the dataset
df = pd.read_csv("mental_health.csv")

# Data Cleaning and Pre-processing

# Drop null values
df.dropna(inplace=True)

#Feature Extraction: Keep only 'post_text' and 'label' columns
data = df[["post_text", "label"]]


# Preprocess function
def preprocess_text(text):
    """
    Preprocessing the data
    Removing numbers
    Removing special characters such as @ #
    Converting to lowercase
    Tokenization
    Removing stopwords
    """

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Join the words back into a string
    text = " ".join(words)

    return text


# Apply preprocessing to the post_text column
data["processed_text"] = data["post_text"].apply(preprocess_text)

# Drop rows with empty strings in the "processed_text" column
data = data[data["processed_text"].astype(bool)]

# Display the processed data
print(data[["post_text", "processed_text", "label"]].head())

# Save the preprocessed data to a new CSV file
data[["processed_text", "label"]].to_csv("processed_data.csv", index=False)
