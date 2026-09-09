"""Embed the synthetic training-only resumes into a dedicated Qdrant collection.

Deliberately separate from src/ingest.py's collection: that one (config.COLLECTION_NAME,
"hr_copilot_resumes") is what the live app and the n8n flow search at inference time. This script
writes to TRAINING_COLLECTION instead, so the 11 candidates in training/synthetic_resumes/ are
reachable for computing the `similarity` feature during training but are never returned to a real
screening run. Nothing in data/resumes/ or the production collection is read or touched.

Reuses parse_resume from src/ingest.py rather than re-implementing the fixed-layout parsing, so
there is exactly one definition of how a CV PDF becomes candidate metadata.

Run from the repo root with a Python that has the project's requirements.txt installed:
    python training/ingest_synthetic.py [--reset]
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
SYNTHETIC_RESUMES_DIR = TRAINING_DIR / "synthetic_resumes"

for _src in (PROJECT_ROOT / "src", Path("/app/src")):
    if (_src / "ingest.py").exists():
        sys.path.insert(0, str(_src))
        break
else:
    sys.exit("could not locate the project's src/ directory")

# Must run before importing config: config.py reads QDRANT_URL etc. from the environment at
# import time, so loading .env any later would leave it on the localhost default and every call
# would refuse to connect.
load_dotenv(PROJECT_ROOT / ".env")

from config import EMBEDDING_MODEL, QDRANT_API_KEY, QDRANT_URL  # noqa: E402
from ingest import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIM, load_resumes, wait_for_qdrant  # noqa: E402
from langchain_openai import OpenAIEmbeddings  # noqa: E402

# A name distinct from config.COLLECTION_NAME on purpose - see module docstring.
TRAINING_COLLECTION = "hr_copilot_resumes_training"


def ensure_training_collection(client: QdrantClient, reset: bool) -> None:
    exists = client.collection_exists(TRAINING_COLLECTION)
    if exists and reset:
        print(f"Dropping existing collection '{TRAINING_COLLECTION}'...")
        client.delete_collection(TRAINING_COLLECTION)
        exists = False
    if not exists:
        print(f"Creating collection '{TRAINING_COLLECTION}' ({EMBEDDING_DIM}-dim, cosine)...")
        client.create_collection(
            collection_name=TRAINING_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{TRAINING_COLLECTION}' already exists - adding to it.")
    client.create_payload_index(
        collection_name=TRAINING_COLLECTION,
        field_name="metadata.years_experience",
        field_schema=PayloadSchemaType.INTEGER,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest synthetic training resumes into Qdrant.")
    parser.add_argument("--reset", action="store_true", help="Drop the training collection first.")
    args = parser.parse_args()

    print(f"Reading synthetic resumes from {SYNTHETIC_RESUMES_DIR}")
    documents = load_resumes(SYNTHETIC_RESUMES_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"\nSplit {len(documents)} resumes into {len(chunks)} chunks")

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        wait_for_qdrant(client)
        ensure_training_collection(client, reset=args.reset)

        vector_store = QdrantVectorStore(
            client=client, collection_name=TRAINING_COLLECTION, embedding=embeddings
        )
        print(f"Embedding with '{EMBEDDING_MODEL}' and writing to {TRAINING_COLLECTION}...")
        vector_store.add_documents(chunks)

        count = client.count(collection_name=TRAINING_COLLECTION).count
        print(f"\nDone. Collection '{TRAINING_COLLECTION}' now holds {count} points.")
    except Exception as error:
        print(f"\n[!] Ingestion failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
