from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
}


def load_document(file_path: Path):
    """Load one supported document and preserve its metadata."""

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(file_path))

    elif extension in {".txt", ".md"}:
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    elif extension == ".csv":
        loader = CSVLoader(
            str(file_path),
            encoding="utf-8",
        )

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = file_path.name
        document.metadata["source_path"] = str(file_path)
        document.metadata["file_type"] = extension

    return documents


def load_documents(data_dir: str = "app/data"):
    """Recursively load all supported documents."""

    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {directory}"
        )

    files = sorted(
        file
        for file in directory.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"No supported documents found in {directory}")
        return []

    all_documents = []

    for file_path in files:
        try:
            documents = load_document(file_path)
            all_documents.extend(documents)

            print(
                f"Loaded {file_path.name}: "
                f"{len(documents)} document sections"
            )

        except Exception as error:
            print(
                f"Failed to load {file_path.name}: "
                f"{type(error).__name__}: {error}"
            )

    print(f"\nTotal loaded sections: {len(all_documents)}")

    return all_documents


if __name__ == "__main__":
    documents = load_documents("app/data")
    print(f"Loaded {len(documents)} documents.")