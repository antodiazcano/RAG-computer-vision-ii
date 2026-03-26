"""
Script to define the class to process LaTeX documents before saving them into the
vector db.
"""

import re
from pathlib import Path

from src.embed_documents.base_processor import Processor


class TEXProcessor(Processor):
    """
    Class that defines the processor to process LaTeX documents before saving them into
    the vector db. It chunks by paragraph and tracks section/subsection numbering.
    """

    @staticmethod
    def _strip_latex(text: str) -> str:
        """
        Strips LaTeX commands from text while keeping raw math expressions.

        Args:
            text: Raw LaTeX text.

        Returns:
            Cleaned text.
        """

        # Remove comments
        text = re.sub(r"(?<!\\)%.*", "", text)
        # Remove \begin{...} and \end{...}
        text = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", text)
        # Remove \label{...}, \ref{...}, \cite{...}, \usepackage{...}, etc.
        text = re.sub(
            r"\\(?:label|ref|cite|usepackage|documentclass)\{[^}]*\}", "", text
        )
        # Unwrap commands like \textbf{content} -> content
        text = re.sub(
            r"\\(?:textbf|textit|emph|underline|text)\{([^}]*)\}", r"\1", text
        )
        # Remove remaining commands without braces (e.g., \maketitle, \newpage)
        text = re.sub(r"\\(?:maketitle|newpage|tableofcontents|noindent)\b", "", text)
        # Clean up braces left over
        text = text.replace("{", "").replace("}", "")
        # Collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def _extract_body(content: str) -> str:
        """
        Extracts the document body, removing the preamble and end{document}.

        Args:
            content: Full LaTeX file content.

        Returns:
            Content between begin{document} and end{document}.
        """

        doc_match = re.search(r"\\begin\{document\}", content)
        if doc_match:
            content = content[doc_match.end() :]
        content = re.sub(r"\\end\{document\}.*", "", content, flags=re.DOTALL)

        return content

    @staticmethod
    def _split_into_sections(content: str) -> tuple[list[tuple[str, str]], int]:
        """
        Splits the document body into sections, tracking section/subsection numbering.

        Args:
            content: Document body (after preamble removal).

        Returns:
            - A list of (section_id, raw_text) pairs.
            - The total number of top-level sections.
        """

        section_pattern = re.compile(r"\\(section|subsection)\{([^}]*)\}")

        sections: list[tuple[str, str]] = []
        sec_num = 0
        subsec_num = 0
        last_pos = 0
        current_section_id = "0"

        for match in section_pattern.finditer(content):
            text_before = content[last_pos : match.start()]
            if text_before.strip():
                sections.append((current_section_id, text_before))

            if match.group(1) == "section":
                sec_num += 1
                subsec_num = 0
                current_section_id = str(sec_num)
            else:
                subsec_num += 1
                current_section_id = f"{sec_num}.{subsec_num}"

            last_pos = match.end()

        remaining = content[last_pos:]
        if remaining.strip():
            sections.append((current_section_id, remaining))

        return sections, sec_num

    def _obtain_chunks(self) -> list[dict[str, str | int]]:
        """
        Obtains the chunks from a LaTeX file. One chunk per paragraph, tracking
        section and subsection numbering.

        Returns:
            A list that for each chunk contains:
                - The text of the chunk.
                - The file where the chunk was retrieved.
                - The location of the retrieved chunk (section id).
                - The total number of locations in the file (total sections).
                - The document type of the file of the chunk.
        """

        content = Path(self.path).read_text(encoding="utf-8")
        body = self._extract_body(content)
        sections, total_sections = self._split_into_sections(body)

        chunks: list[dict[str, str | int]] = []
        for section_id, raw_text in sections:
            cleaned = self._strip_latex(raw_text)
            paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

            for paragraph in paragraphs:
                chunks.append(
                    {
                        "text": paragraph,
                        "source": self.path.name,
                        "location": section_id,
                        "total_locations": total_sections,
                        "doc_type": "tex",
                    }
                )

        return chunks
