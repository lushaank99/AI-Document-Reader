import argparse
import hashlib
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

DB_PATH = os.environ.get("DOCMGR_DB", "documents.db")
MODEL = os.environ.get("DOCMGR_MODEL", "claude-sonnet-5-5")
CHUNK_SIZE = 900      
CHUNK_OVERLAP = 150   
TOP_K = 5             

STOPWORDS = set(
    "a an and are as at be but by for from has have he her his i if in into is it its "
    "of on or she that the their them they this to was we were what when where which "
    "who will with you your do does did not no can could should would about how".split()
)


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT NOT NULL,
            sha256   TEXT NOT NULL UNIQUE,
            pages    INTEGER NOT NULL,
            added_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page   INTEGER NOT NULL,
            text   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        """
    )
    return conn


def extract_pages(pdf_path):
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("Missing dependency. Run: pip install pypdf")

    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("PDF is password-protected.")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if text:
            pages.append((number, text))
    return pages, len(reader.pages)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > size * 0.5:
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def add_pdf(conn, pdf_path):
    path = Path(pdf_path)
    if not path.is_file() or path.suffix.lower() != ".pdf":
        print(f"  ! Skipping '{pdf_path}': not a PDF file.")
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    existing = conn.execute("SELECT name FROM documents WHERE sha256 = ?", (digest,)).fetchone()
    if existing:
        print(f"  = '{path.name}' is already stored (as '{existing[0]}').")
        return

    try:
        pages, total_pages = extract_pages(str(path))
    except Exception as exc:
        print(f"  ! Could not read '{path.name}': {exc}")
        return
    if not pages:
        print(f"  ! '{path.name}' has no extractable text (scanned images? try OCR first).")
        return

    cur = conn.execute(
        "INSERT INTO documents (name, sha256, pages, added_at) VALUES (?, ?, ?, ?)",
        (path.name, digest, total_pages, datetime.now().isoformat(timespec="seconds")),
    )
    doc_id = cur.lastrowid
    count = 0
    for page_no, text in pages:
        for piece in chunk_text(text):
            conn.execute(
                "INSERT INTO chunks (doc_id, page, text) VALUES (?, ?, ?)",
                (doc_id, page_no, piece),
            )
            count += 1
    conn.commit()
    print(f"  + Stored '{path.name}': {total_pages} pages, {count} chunks.")



def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def search(conn, question, k=TOP_K, doc_name=None):
    sql = (
        "SELECT c.id, d.name, c.page, c.text FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id"
    )
    params = ()
    if doc_name:
        sql += " WHERE d.name = ?"
        params = (doc_name,)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []

    q_terms = set(tokenize(question))
    if not q_terms:
        return []

    docs = [Counter(tokenize(r[3])) for r in rows]
    lengths = [sum(d.values()) for d in docs]
    avg_len = (sum(lengths) / len(lengths)) or 1
    n = len(rows)
    df = {t: sum(1 for d in docs if t in d) for t in q_terms}

    k1, b = 1.5, 0.75
    scored = []
    for row, tf, length in zip(rows, docs, lengths):
        score = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            freq = tf[t]
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * length / avg_len))
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"doc": r[1], "page": r[2], "text": r[3], "score": s} for s, r in scored[:k]
    ]


SYSTEM_PROMPT = (
    "You answer questions using ONLY the document excerpts provided. "
    "Cite sources as (filename, p.N). If the excerpts do not contain the answer, "
    "say you could not find it in the documents. Do not invent facts."
)


def answer_with_claude(question, hits):
    try:
        import anthropic
    except ImportError:
        return None
    context = "\n\n".join(
        f"[{h['doc']} | page {h['page']}]\n{h['text']}" for h in hits
    )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def ask(conn, question, doc_name=None):
    hits = search(conn, question, doc_name=doc_name)
    if not hits:
        return "I couldn't find anything relevant in the stored documents."

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            answer = answer_with_claude(question, hits)
            if answer:
                return answer
        except Exception as exc:
            print(f"  (AI answering failed: {exc}; showing best passages instead)\n")

    lines = ["Best matching passages (set ANTHROPIC_API_KEY for written answers):\n"]
    for i, h in enumerate(hits, 1):
        snippet = h["text"].replace("\n", " ")
        lines.append(f"{i}. [{h['doc']}, p.{h['page']}] {snippet}\n")
    return "\n".join(lines)


def list_docs(conn):
    rows = conn.execute(
        "SELECT d.name, d.pages, d.added_at, COUNT(c.id) FROM documents d "
        "LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.id ORDER BY d.added_at"
    ).fetchall()
    if not rows:
        print("No documents stored yet. Add one with: add <file.pdf>")
        return
    print(f"{'Name':40} {'Pages':>5} {'Chunks':>7}  Added")
    for name, pages, added, chunks in rows:
        print(f"{name[:40]:40} {pages:>5} {chunks:>7}  {added}")


def delete_doc(conn, name):
    cur = conn.execute("DELETE FROM documents WHERE name = ?", (name,))
    conn.commit()
    print(f"Deleted '{name}'." if cur.rowcount else f"No document named '{name}'.")


def chat(conn):
    print("Ask questions about your documents. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            print(f"\n{ask(conn, question)}\n")


def main():
    parser = argparse.ArgumentParser(description="AI Document Manager for PDFs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add one or more PDF files")
    p_add.add_argument("files", nargs="+")

    sub.add_parser("list", help="List stored documents")

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--doc", help="Limit the search to one stored document name")

    sub.add_parser("chat", help="Interactive question mode")

    p_del = sub.add_parser("delete", help="Remove a stored document")
    p_del.add_argument("name")

    args = parser.parse_args()
    conn = connect()

    if args.command == "add":
        for f in args.files:
            add_pdf(conn, f)
    elif args.command == "list":
        list_docs(conn)
    elif args.command == "ask":
        print(ask(conn, args.question, args.doc))
    elif args.command == "chat":
        chat(conn)
    elif args.command == "delete":
        delete_doc(conn, args.name)


if __name__ == "__main__":
    main()