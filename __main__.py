from enum import Enum, IntEnum

from langchain_core.vectorstores import InMemoryVectorStore

from src.radiohead_rag.embeddings import parse_csv_to_vector_store
from src.radiohead_rag.retrieve_dataset import retrieve_dataset_from_kaggle

DATASET_ID = "meetshahnuv/radiohead-song-lyrics"
DATASET_CSV = "data/radiohead.csv"


class Columns(str, Enum):
    LYRICS = "Lyrics"
    ALBUM = "Album Name"
    TRACK = "Track Name"


class Chunk(IntEnum):
    SIZE = 200
    OVERLAP = 50


def main():
    retrieve_dataset_from_kaggle(
        dataset_id=DATASET_ID,
        target_path=DATASET_CSV,
    )
    vector_store: InMemoryVectorStore = parse_csv_to_vector_store(
        csv_path=DATASET_CSV,
        source_column=Columns.LYRICS,
        metadata_columns=(Columns.TRACK, Columns.TRACK),
        chunk_size=Chunk.SIZE,
        chunk_overlap=Chunk.OVERLAP,
    )
    print(vector_store)


if __name__ == "__main__":
    main()
