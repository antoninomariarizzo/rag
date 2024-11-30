import torch
import faiss
from typing import List, Tuple
from src.LargeLanguageModel import LargeLanguageModel


class IndexManager:
    """
    Handles creating, managing, and querying a FAISS index.
    """

    def __init__(self,
                 texts: List[str],
                 info: List[str],
                 llm: LargeLanguageModel):
        """
        Initialize the Index Manager.

        Parameters:
        - texts: List of text chunks corresponding to embeddings.
        - info: Additional info (e.g., PDF filenames and page numbers 
        where the text has been extracted).
        - llm: Model used to extract the embeddings.
        """
        self.index = None  # FAISS index instance.
        self.texts = texts
        self.info = info  # metadata for the text chunks.
        self.llm = llm

    def create_index(self,
                     texts: List[str],
                     out_path: str = None) -> None:
        """
        Create a FAISS index for the provided texts.

        Parameters:
        - texts: List of text chunks.
        - out_path: Path to save the FAISS index.

        Raises:
        - ValueError: If the embeddings are not 2D or are of invalid shape.
        """
        embeddings = self.llm.encode(texts)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D tensor of shape (n_samples, embedding_dim).")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Save index if a path is provided
        if out_path is not None:
            self.save_index(out_path)

    def save_index(self, out_path: str) -> None:
        """
        Save the FAISS index to a file.

        Parameters:
        - out_path: File path to save the index.

        Raises:
        - ValueError: If the index has not been created.
        """
        if self.index is None:
            raise ValueError("Index has not been created yet.")
        faiss.write_index(self.index, out_path)

    def load_index(self, in_path: str) -> None:
        """
        Load a FAISS index from a file.

        Parameters:
        - in_path: File path to load the index.
        """
        self.index = faiss.read_index(in_path)

    def query(self,
              text: str,
              top_k: int = 5) -> List[Tuple[str, int]]:
        """
        Query the FAISS index for relevant text chunks.

        Parameters:
        - text: Query text string.
        - top_k: Number of top results to retrieve.

        Returns:
        - references: Retrieved text chunks, along with 
        their info (such as pdf filename and page number).
        Raises:
        - ValueError: If the index has not been created.
        """
        if self.index is None:
            raise ValueError("Index has not been created yet.")

        query_embedding = self.llm.encode(text)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        # Add batch dimension if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[None, :]

        # Perform Similarity Search
        D, I = self.index.search(query_embedding, k=top_k)

        # Since `I` contains the indices of the text chunks related to the query, 
        # we use these indices to retrieve both the text and the 
        # additional information (such as PDF filenames and page numbers)
        references = []
        for idx in I[0]:
            if idx == -1:
                continue

            references.append((self.texts[idx], self.info[idx]))

        return references

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
        """
        Splits a text into smaller chunks of a maximum of `chunk_size` words.

        Parameters:
        - text: The text to be split into chunks.
        - chunk_size: Number of maximum words in the chunk

        Returns:
        - chunks: List of text chunks, each containing up to `chunk_size` words.
        """
        words = text.split()
        chunks = [" ".join(words[i:i + chunk_size])
                  for i in range(0, len(words), chunk_size)]
        return chunks

    @staticmethod
    def chunk_texts_and_info(texts: List[str],
                             info: List[str],
                             chunk_size: int = 200
                             ) -> Tuple[List[str]]:
        """
        Splits a list of texts into smaller chunks of a maximum of `chunk_size` words.
        For each chunk of text, the corresponding info (e.g., PDF filenames and page numbers)
        is repeated, maintaining a mapping between chunks and info.

        Parameters:
        - texts: List of text strings to be split into chunks.
        - info: List of info corresponding to the texts (e.g., PDF filenames and page numbers).
        - chunk_size: Maximum number of words in each chunk.

        Returns:
        - out_chunks: List of text chunks.
        - out_info: List of info, repeated for each text chunk.
        """
        out_info = []
        out_chunks = []

        for text_cur, info_cur, in zip(texts, info):
            # Split current text into chunks
            chunks = IndexManager.chunk_text(text_cur, chunk_size)

            # Repeat info_cur for each element in the chunk 
            # to mantain a mapping between chunks and info
            num_chunks = len(chunks)

            out_info.extend([info_cur] * num_chunks)
            out_chunks.extend(chunks)

        return out_chunks, out_info
