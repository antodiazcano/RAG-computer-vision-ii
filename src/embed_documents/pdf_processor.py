"""
Script to define the class to process PDF documents before saving them into the vector
db.
"""

import fitz

from src.embed_documents.base_processor import Processor


class PDFProcessor(Processor):
    """
    Class that defines the processor to process PDF documents before saving them into
    the vector db.  It chunks by page and tracks page numbering.
    """

    def _obtain_chunks(self) -> list[dict[str, str | int]]:
        """
        Obtains the chunks from one file.

        Returns:
            A list that for each chunk contains:
                - The text of the chunk.
                - The file where the chunk was retrieved.
                - The page of the retrieved chunk.
                - The total number of pages of the file of the chunk.
                - The document type of the file of the chunk.
        """

        doc = fitz.open(str(self.path))
        chunks = []
        total_pages = len(doc)

        for page_num in range(total_pages):
            print(f"Obtaining chunks for page {page_num}...")
            page = doc[page_num]
            text = page.get_text("text").strip()
            if not text:
                print(
                    f"Could not extract text from page {page_num} of file "
                    f"{self.path}."
                )
                continue
            chunks.append(
                {
                    "text": text,
                    "source": self.path.name,
                    "page": page_num + 1,
                    "total_pages": total_pages,
                    "doc_type": "pdf",
                }
            )

        doc.close()

        return chunks
