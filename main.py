import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(url="http://localhost:6333/")

groq_client = Groq(api_key=GROQ_API_KEY)

with open("data/resume.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = text.split("\n\n")

embeddings = model.encode(chunks)

client.recreate_collection(
    collection_name="resume_docs",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

points = []

for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id=i,
        vector=embedding.tolist(),
        payload={
            "text": chunks[i]
        }
    )
    points.append(point)

client.upsert(
    collection_name="resume_docs",
    points=points
)

question = input("Ask a question about my resume: ")

question_embedding = model.encode(question)

results = client.query_points(
    collection_name="resume_docs",
    query=question_embedding.tolist(),
    limit=3,
    with_payload=True
).points

context = ""

for result in results:
    context += result.payload["text"] + "\n"

response = groq_client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": """Answer the question using only the information
provided in the resume context.
Do not make up information.
If the answer is not present in the resume,
say "I don't know based on the provided resume."
"""
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]
)

answer = response.choices[0].message.content

print("\nAnswer:")
print(answer)