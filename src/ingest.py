"""
HR Copilot - RAG ingestion.

Reads the synthetic candidate resumes from data/resumes/, splits them into chunks,
embeds them with OpenAI's text-embedding-3-small, and stores them in the Qdrant
collection used by the CV Retrieval node (Node 2) of the LangGraph flow.

Chunking strategy (documented in system_specification.pdf):
    Resumes are short, self-contained documents, so we use a RecursiveCharacterTextSplitter
    with a chunk size large enough that most resumes stay in 1-2 chunks. Splitting mainly
    protects us from an unusually long CV. Every chunk carries the *same* candidate metadata
    (name, role, years of experience, skills) plus the FULL resume text, so that the Evaluator
    node (Node 3) always scores a complete CV rather than a fragment that happened to match.

Run it once before using the graph:
    python src/ingest.py            # add the resumes to the collection
    python src/ingest.py --reset    # drop and rebuild the collection first
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from config import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_API_KEY, QDRANT_URL, RESUMES_DIR

load_dotenv()

# text-embedding-3-small produces 1536-dimensional vectors.
EMBEDDING_DIM = 1536

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def extract_pdf_text(path: Path) -> str:
    """Extract plain text from a PDF resume, one page's text per line group."""
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() for page in reader.pages)


def parse_resume(path: Path) -> Document:
    """Turn one PDF resume into a Document with structured metadata.

    The synthetic resumes follow a fixed layout (name on line 1, role on line 2, plus
    'Skills:' and 'Years of experience:' lines), so once the PDF text is extracted, the
    metadata is parsed deterministically instead of with an LLM call - it is cheaper,
    reproducible, and easy to explain.
    """
    text = extract_pdf_text(path).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    candidate_name = lines[0] if lines else path.stem
    role_title = lines[1] if len(lines) > 1 else "Unknown"

    skills: list[str] = []
    years_experience = 0
    email = ""

    for line in lines:
        if line.lower().startswith("email:"):
            email = line.split(":", 1)[1].strip()
        elif line.lower().startswith("skills:"):
            raw_skills = line.split(":", 1)[1]
            skills = [s.strip() for s in raw_skills.split(",") if s.strip()]
        elif line.lower().startswith("years of experience:"):
            raw_years = line.split(":", 1)[1].strip()
            try:
                years_experience = int(raw_years)
            except ValueError:
                print(f"  [!] Could not parse years of experience in {path.name}: {raw_years!r}")

    return Document(
        page_content=text,
        metadata={
            "candidate_name": candidate_name,
            "role_title": role_title,
            "years_experience": years_experience,
            "skills": skills,
            # Needed by Node 4 to address the interview invitation.
            "email": email,
            "source_file": path.name,
            # The complete CV, carried on every chunk so the Evaluator node never
            # has to score a partial resume.
            "full_text": text,
        },
    )


def load_resumes(resumes_dir: Path) -> list[Document]:
    """Load every resume in the directory as a Document."""
    if not resumes_dir.is_dir():
        raise FileNotFoundError(f"Resumes directory not found: {resumes_dir}")

    resume_paths = sorted(resumes_dir.glob("*.pdf"))
    if not resume_paths:
        raise FileNotFoundError(f"No .pdf resumes found in {resumes_dir}")

    documents = []
    for path in resume_paths:
        document = parse_resume(path)
        documents.append(document)
        print(
            f"  Loaded {path.name}: {document.metadata['candidate_name']} "
            f"({document.metadata['role_title']}, {document.metadata['years_experience']}y, "
            f"{len(document.metadata['skills'])} skills)"
        )
    return documents


def ensure_collection(client: QdrantClient, reset: bool) -> None:
    """Create the Qdrant collection if needed, optionally dropping the existing one."""
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and reset:
        print(f"Dropping existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        print(f"Creating collection '{COLLECTION_NAME}' ({EMBEDDING_DIM}-dim, cosine)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists - adding to it.")

    # Qdrant Cloud rejects a range filter on an unindexed field, and retrieval.py filters on
    # years of experience. Without this, every requisition asking for N+ years returns nothing -
    # silently, because cv_retrieval treats a failed search as zero candidates.
    print("Ensuring payload index on 'metadata.years_experience'...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.years_experience",
        field_schema=PayloadSchemaType.INTEGER,
    )


def wait_for_qdrant(client: QdrantClient, attempts: int = 30, delay_seconds: float = 2.0) -> None:
    """Retry until Qdrant answers, so this can run as a Docker Compose init step that starts
    as soon as the qdrant container is created rather than waiting for it to be ready."""
    for attempt in range(1, attempts + 1):
        try:
            client.get_collections()
            return
        except Exception as error:
            if attempt == attempts:
                raise
            print(f"  Qdrant not ready yet ({attempt}/{attempts}): {error}")
            time.sleep(delay_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest candidate resumes into Qdrant.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the collection before ingesting, so re-runs do not duplicate candidates.",
    )
    args = parser.parse_args()

    print(f"Reading resumes from {RESUMES_DIR}")
    documents = load_resumes(RESUMES_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"\nSplit {len(documents)} resumes into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        wait_for_qdrant(client)
        ensure_collection(client, reset=args.reset)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )

        print(f"Embedding with '{EMBEDDING_MODEL}' and writing to {QDRANT_URL}...")
        vector_store.add_documents(chunks)

        count = client.count(collection_name=COLLECTION_NAME).count
        print(f"\nDone. Collection '{COLLECTION_NAME}' now holds {count} points.")
    except Exception as error:
        # The usual cause is Qdrant not running yet (docker compose up -d qdrant) or a
        # missing/invalid OPENAI_API_KEY.
        print(f"\n[!] Ingestion failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
