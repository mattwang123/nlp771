import json
import pickle
import numpy as np
import faiss

# -----------------------
# Load documents
# -----------------------
with open("data/corpus.jsonl", "r") as f:
    docs = [json.loads(line) for line in f]
print(f"Loaded {len(docs)} documents from corpus.")

# Build doc_id → index mapping
doc_ids = [d["doc_id"] for d in docs]

# -----------------------
# Load document embeddings
# -----------------------
with open("scifact_evidence_embeddings.pkl", "rb") as f:
    doc_embeddings = pickle.load(f)

# Keep only documents with embeddings
doc_vectors = []
filtered_doc_ids = []
for (doc_id, abstract), embedding in doc_embeddings.items():
    filtered_doc_ids.append(doc_id)
    doc_vectors.append(embedding)

doc_vectors = np.vstack(doc_vectors)
print(f"Prepared {len(filtered_doc_ids)} document embeddings for FAISS index.")

# -----------------------
# Load claims
# -----------------------
with open("data/claims_train.jsonl", "r") as f:
    claims = [json.loads(line) for line in f]

# -----------------------
# Load claim embeddings
# -----------------------
with open("scifact_claim_embeddings.pkl", "rb") as f:
    claim_embeddings = pickle.load(f)

# -----------------------
# Align claims with embeddings & ground truth
# -----------------------
claim_vectors = []
ground_truth_doc_ids = []

for claim in claims:
    key = (claim["id"], claim["claim"])  # keys in embedding pickle
    cited = claim.get("cited_doc_ids", [])
    if cited and key in claim_embeddings:
        claim_vectors.append(claim_embeddings[key])
        ground_truth_doc_ids.append(set(str(cid) for cid in cited))

if not claim_vectors:
    raise RuntimeError("No claim embeddings match the claims file — check pickle keys!")

claim_vectors = np.vstack(claim_vectors)
print(f"Prepared {len(claim_vectors)} claim embeddings with ground truth.")

# -----------------------
# Build FAISS index
# -----------------------
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)  # inner product (cosine similarity if normalized)
index.add(doc_vectors.astype("float32"))
print(f"Built FAISS index with {len(filtered_doc_ids)} documents (dim={dimension}).")

# -----------------------
# Define metrics
# -----------------------
def calculate_metrics(retrieved_docs, ground_truth_docs, k):
    mrr_scores = []
    map_scores = []

    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        # MRR
        rr = 0
        for i, doc_id in enumerate(retrieved[:k]):
            if str(doc_id) in truth:
                rr = 1.0 / (i + 1)
                break
        mrr_scores.append(rr)

        # MAP
        relevant_found = 0
        precision_sum = 0
        for i, doc_id in enumerate(retrieved[:k]):
            if str(doc_id) in truth:
                relevant_found += 1
                precision_sum += relevant_found / (i + 1)
        ap = precision_sum / len(truth) if truth else 0
        map_scores.append(ap)

    return np.mean(mrr_scores), np.mean(map_scores)

# -----------------------
# Retrieval and evaluation
# -----------------------
k_values = [1, 10, 50]
results = {}

print("\nEvaluating IR system performance:")
# Search with maximum k to get top-50 results in one call
scores, indices = index.search(claim_vectors.astype("float32"), max(k_values))

for k in k_values:
    retrieved_docs = []
    for idx_list in indices[:, :k]:
        retrieved_docs.append([filtered_doc_ids[i] for i in idx_list])

    mrr, map_score = calculate_metrics(retrieved_docs, ground_truth_doc_ids, k)
    results[k] = {"MRR": mrr, "MAP": map_score}
    print(f"k={k}: MRR={mrr:.4f}, MAP={map_score:.4f}")

# -----------------------
# Print LaTeX table
# -----------------------
print("\nResults for LaTeX Table:")
print(
    f"OpenAI Embeddings & {results[1]['MRR']:.4f} & {results[1]['MAP']:.4f} & "
    f"{results[10]['MRR']:.4f} & {results[10]['MAP']:.4f} & "
    f"{results[50]['MRR']:.4f} & {results[50]['MAP']:.4f} \\\\"
)
