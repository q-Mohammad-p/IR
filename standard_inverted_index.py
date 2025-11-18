import os
import spacy
from nltk.stem import PorterStemmer
from BTrees.OOBTree import OOBTree  # B-Tree واقعی


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

ps = PorterStemmer()


sample_files = ["file1.txt", "file2.txt", "file3.txt"]


def process_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = []

    for token in doc:
        if not token.is_alpha:
            continue
        if token.is_stop:
            continue

        lemma = token.lemma_
        stem = ps.stem(lemma)
        tokens.append(stem)

    return tokens


def build_inverted(files):
    index = {}

    for doc_id, fname in enumerate(files, start=1):
        with open(fname, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = process_text(text)
        for tok in tokens:
            if tok not in index:
                index[tok] = {}

            index[tok][doc_id] = 1

    return index


def store_btree(inverted):
    tree = OOBTree()
    for term, postings in inverted.items():
        tree[term] = postings
    return tree


def print_inverted(inv):
    print("\n STANDARD INVERTED INDEX : \n")
    for term in sorted(inv.keys()):
        posting = inv[term]
        files = ", ".join([f"file{d}.txt" for d in sorted(posting.keys())])
        print(f"{term} → {files}")
    print()


def search_menu(tree):
    print("\n=== Search Engine ===")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Search term: ").strip().lower()
        if q == "exit":
            break

        doc = nlp(q)
        lemma = doc[0].lemma_
        stem = ps.stem(lemma)

        result = tree.get(stem, None)

        if result:
            files = ", ".join([f"file{doc_id}.txt" for doc_id in sorted(result.keys())])
            print(f"Found in: {files}\n")
        else:
            print(f"'{q}' not found.\n")


if __name__ == "__main__":
    inverted = build_inverted(sample_files)
    print_inverted(inverted)

    btree = store_btree(inverted)
    search_menu(btree)
