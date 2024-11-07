from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import numpy as np

load_dotenv()

# Load the embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()


# Example: Extracting ideas and descriptions
data = {
    "type": "ISSUE",
    "id": "E-1T4C6D9-133",
    "children": [
        {
            "type": "CRITERION",
            "name": "high impact",
            "description": "substantial impact",
        },
        {
            "type": "IDEA",
            "name": "Promote studies",
            "description": "Encourage ADDLabs members",
        },
        # Add more children nodes here
    ],
}

# Step 1: Generate embeddings for each "IDEA" or "CRITERION" node
node_embeddings = []
node_texts = []
for child in data["children"]:
    text = f"{child['name']} - {child['description']}"
    embedding = get_embedding(text)
    node_embeddings.append(embedding)
    node_texts.append(text)

# Convert embeddings list to numpy array
node_embeddings = np.vstack(node_embeddings)

# Step 2: Perform PCA to reduce dimensions for KMeans
pca = PCA(n_components=10)  # Adjust based on data size
reduced_embeddings = pca.fit_transform(node_embeddings)


# Step 3: Perform divisive clustering recursively
def divisive_clustering(embeddings, depth=0, max_depth=3):
    if depth >= max_depth or len(embeddings) <= 2:
        return [embeddings]

    # KMeans with k=2 to split cluster
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(embeddings)

    # Divide into subclusters
    cluster_1 = embeddings[labels == 0]
    cluster_2 = embeddings[labels == 1]

    return divisive_clustering(cluster_1, depth + 1) + divisive_clustering(
        cluster_2, depth + 1
    )


# Step 4: Run clustering on reduced embeddings
clusters = divisive_clustering(reduced_embeddings)

# Display clusters
for idx, cluster in enumerate(clusters):
    print(f"Cluster {idx + 1}:")
    for item in cluster:
        print(item)  # Summarize or print item for review


from openai import OpenAI
import os

MODEL = "gpt-4o"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
print(client)


MODEL = "gpt-4"


def summarize_cluster(cluster_texts):
    combined_text = " ".join(cluster_texts)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes content concisely to be Mutually Exclusive and Collectively Exhaustive.",
            },
            {
                "role": "user",
                "content": f"Please summarize the following ideas: {combined_text}",
            },
        ],
        max_tokens=100,
    )

    summary = response.choices[0].message.content

    return summary


# Summarize each cluster
for idx, cluster in enumerate(clusters):
    cluster_texts = [
        node_texts[i] for i, item in enumerate(cluster) if item is not None
    ]
    summary = summarize_cluster(cluster_texts)
    print(f"Cluster {idx + 1} Summary: {summary}")
