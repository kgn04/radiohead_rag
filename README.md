# radiohead_rag

Ask queries about Radiohead tracks' lyrics.

## `LangChain` pipeline

### Retrieving dataset from `Kaggle`

Using `kagglehub` Python API.

### Parsing CSV to embeddings vector store

* `CSVLoader` - Retrieving `Documents` from CSV file.
* `RecursiveCharacterTextSplitter` - Chunking lyrics.
* `HuggingFaceEmbeddings` - Embeddings.
