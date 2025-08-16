from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _clear_meta_name(name: str) -> str:
    return name.split(" by")[0]


def _parse_csv_to_documents(
    csv_path: str, source_column: str, metadata_columns: tuple[str]
) -> list[Document]:
    loader = CSVLoader(
        file_path=csv_path,
        source_column=source_column,
        metadata_columns=metadata_columns,
    )
    docs: list[Document] = loader.load()

    for doc in docs:
        for meta_name in metadata_columns:
            doc.metadata[meta_name] = _clear_meta_name(doc.metadata[meta_name])

    return docs


def _chunk_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def _vector_store_from_chunks(chunks: list[Document]) -> InMemoryVectorStore:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = InMemoryVectorStore(embedding_model)
    vector_store.add_documents(chunks)
    return vector_store


def parse_csv_to_vector_store(
    csv_path: str,
    source_column: str,
    metadata_columns: tuple[str],
    chunk_size: int,
    chunk_overlap: int,
) -> InMemoryVectorStore:
    docs: list[Document] = _parse_csv_to_documents(
        csv_path=csv_path,
        source_column=source_column,
        metadata_columns=metadata_columns,
    )
    chunks: list[Document] = _chunk_documents(
        docs=docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return _vector_store_from_chunks(chunks)
