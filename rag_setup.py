import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load dataset
df = pd.read_csv("schemes.csv")

# Convert rows into text format
documents = df.apply(lambda row: " ".join(row.values.astype(str)), axis=1).tolist()

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB
db = FAISS.from_texts(documents, embedding_model)

# Save index
db.save_local("scheme_index")

print("FAISS index created successfully!")
