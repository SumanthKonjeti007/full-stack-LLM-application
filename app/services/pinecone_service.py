import os
import pinecone
from app.services.openai_service import get_embedding

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Define embedding dimension
EMBEDDING_DIMENSION = 768

def embed_chunks_and_upload_to_pinecone(chunks, index_name):
    # Delete the index if it already exists
    if index_name in pc.list_indexes().names():
        pc.delete_index(name=index_name)
    
    # Create a new index in Pinecone
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric='cosine',
        spec=pinecone.ServerlessSpec(cloud='aws', region="us-east-1")
    )
    index = pc.Index(index_name)
    
    # Embed each chunk and aggregate these embeddings
    embeddings_with_ids = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings_with_ids.append((str(i), embedding, chunk))
    
    print(embeddings_with_ids)
    # Upload the embeddings and relevant texts for each chunk to the Pinecone index
    upserts = [(id, vec, {"chunk_text": text}) for id, vec, text in embeddings_with_ids]
    index.upsert(vectors=upserts)

def get_most_similar_chunks_for_query(query, index_name):
    question_embedding = get_embedding(query)
    #print(question_embedding)
    index = pc.Index(index_name)
    query_results = index.query(vector=question_embedding, top_k=3, include_metadata=True)
    context_chunks = [x['metadata']['chunk_text'] for x in query_results['matches']]
    print(context_chunks)
    return context_chunks
