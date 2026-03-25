"""
Tests for src/embed_documents/tex_processor.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


# Mock the module-level clients before importing
with (
    patch("src.utils.get_embedding_client", return_value=MagicMock()),
    patch("src.utils.get_index_vector_db", return_value=MagicMock()),
):
    from src.embed_documents.tex_processor import TEXProcessor


class TestStripLatex:
    """Tests for the TEXProcessor._strip_latex static method."""

    def test_removes_comments(self) -> None:
        """Checks that LaTeX comments are removed."""
        assert TEXProcessor._strip_latex("hello % this is a comment") == "hello"

    def test_removes_begin_end(self) -> None:
        """Checks that begin/end environment commands are removed."""
        text = "\\begin{itemize}\nitem one\n\\end{itemize}"
        result = TEXProcessor._strip_latex(text)
        assert "\\begin" not in result
        assert "\\end" not in result
        assert "item one" in result

    def test_removes_label_ref_cite(self) -> None:
        """Checks that label, ref, and cite commands are removed."""
        text = "See \\ref{fig1} and \\cite{paper1} with \\label{sec1}"
        result = TEXProcessor._strip_latex(text)
        assert "\\ref" not in result
        assert "\\cite" not in result
        assert "\\label" not in result
        assert "See" in result

    def test_unwraps_formatting_commands(self) -> None:
        """Checks that textbf, textit, emph are unwrapped to their content."""
        text = "This is \\textbf{bold} and \\textit{italic} and \\emph{emphasis}"
        result = TEXProcessor._strip_latex(text)
        assert result == "This is bold and italic and emphasis"

    def test_preserves_inline_math(self) -> None:
        """Checks that inline math expressions are preserved."""
        text = "The equation $E = mc^2$ is famous"
        result = TEXProcessor._strip_latex(text)
        assert "$E = mc^2$" in result

    def test_preserves_display_math(self) -> None:
        """Checks that display math expressions are preserved."""
        text = "Consider:\n$$\\int_0^1 x^2 dx$$"
        result = TEXProcessor._strip_latex(text)
        assert "$$\\int_0^1 x^2 dx$$" in result

    def test_removes_maketitle(self) -> None:
        """Checks that standalone commands like maketitle are removed."""
        text = "\\maketitle\nSome content"
        result = TEXProcessor._strip_latex(text)
        assert "\\maketitle" not in result
        assert "Some content" in result

    def test_collapses_whitespace(self) -> None:
        """Checks that excessive whitespace is collapsed."""
        assert TEXProcessor._strip_latex("hello     world") == "hello world"

    def test_collapses_multiple_newlines(self) -> None:
        """Checks that more than two consecutive newlines are collapsed."""
        assert (
            TEXProcessor._strip_latex("para one\n\n\n\n\npara two")
            == "para one\n\npara two"
        )

    def test_empty_input(self) -> None:
        """Checks that empty input returns empty string."""
        assert TEXProcessor._strip_latex("") == ""


class TestExtractBody:
    """Tests for the TEXProcessor._extract_body static method."""

    def test_removes_preamble(self) -> None:
        """Checks that everything before begin document is removed."""
        content = "\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\nBody.\n\\end{document}"
        result = TEXProcessor._extract_body(content)
        assert "documentclass" not in result
        assert "usepackage" not in result
        assert "Body." in result

    def test_removes_end_document(self) -> None:
        """Checks that end document and everything after it is removed."""
        content = "\\begin{document}\nBody.\n\\end{document}\nGarbage after."
        result = TEXProcessor._extract_body(content)
        assert "Body." in result
        assert "Garbage" not in result
        assert "\\end{document}" not in result

    def test_no_begin_document(self) -> None:
        """Checks that content without begin document is returned as-is (minus end)."""
        content = "Some raw content.\n\\end{document}"
        result = TEXProcessor._extract_body(content)
        assert "Some raw content." in result

    def test_no_end_document(self) -> None:
        """Checks that content without end document is returned after preamble removal."""
        content = "\\begin{document}\nBody without end."
        result = TEXProcessor._extract_body(content)
        assert "Body without end." in result

    def test_empty_body(self) -> None:
        """Checks that an empty body between begin and end returns whitespace/empty."""
        content = "\\begin{document}\n\\end{document}"
        result = TEXProcessor._extract_body(content)
        assert result.strip() == ""


class TestSplitIntoSections:
    """Tests for the TEXProcessor._split_into_sections static method."""

    def test_single_section(self) -> None:
        """Checks that a single section is parsed correctly."""
        content = "\n\\section{Intro}\nSome text.\n"
        sections, total = TEXProcessor._split_into_sections(content)

        assert total == 1
        assert len(sections) == 1
        assert sections[0][0] == "1"
        assert "Some text." in sections[0][1]

    def test_multiple_sections(self) -> None:
        """Checks that multiple sections are numbered sequentially."""
        content = (
            "\n\\section{A}\nText A.\n\\section{B}\nText B.\n\\section{C}\nText C.\n"
        )
        sections, total = TEXProcessor._split_into_sections(content)

        assert total == 3
        assert len(sections) == 3
        assert [s[0] for s in sections] == ["1", "2", "3"]

    def test_subsections(self) -> None:
        """Checks that subsections are numbered as section.subsection."""
        content = "\n\\section{Main}\nIntro.\n\\subsection{A}\nSub A.\n\\subsection{B}\nSub B.\n"
        sections, total = TEXProcessor._split_into_sections(content)

        assert total == 1
        ids = [s[0] for s in sections]
        assert ids == ["1", "1.1", "1.2"]

    def test_subsection_resets_on_new_section(self) -> None:
        """Checks that subsection numbering resets when a new section starts."""
        content = (
            "\n\\section{One}\n\\subsection{A}\nText.\n"
            "\\section{Two}\n\\subsection{B}\nText.\n"
        )
        sections, total = TEXProcessor._split_into_sections(content)

        assert total == 2
        ids = [s[0] for s in sections]
        assert "1.1" in ids
        assert "2.1" in ids

    def test_text_before_first_section(self) -> None:
        """Checks that text before the first section is captured under id 0."""
        content = "\nPreamble text.\n\\section{First}\nSection text.\n"
        sections, _ = TEXProcessor._split_into_sections(content)

        assert sections[0][0] == "0"
        assert "Preamble text." in sections[0][1]
        assert sections[1][0] == "1"

    def test_no_sections(self) -> None:
        """Checks that content without sections returns one entry under id 0."""
        content = "\nJust plain text.\n"
        sections, total = TEXProcessor._split_into_sections(content)

        assert total == 0
        assert len(sections) == 1
        assert sections[0][0] == "0"

    def test_empty_content(self) -> None:
        """Checks that empty content returns no sections."""
        sections, total = TEXProcessor._split_into_sections("")

        assert total == 0
        assert isinstance(sections, list) and len(sections) == 0

    def test_ignores_empty_text_between_headings(self) -> None:
        """Checks that empty text between consecutive headings is not captured."""
        content = "\n\\section{A}\n\\section{B}\nText B.\n"
        sections, total = TEXProcessor._split_into_sections(content)

        assert total == 2
        assert len(sections) == 1
        assert sections[0][0] == "2"


class TestObtainChunks:
    """Integration tests for the TEXProcessor._obtain_chunks method."""

    def _write_tex(self, tmp_path: Path, content: str) -> Path:
        """Writes a .tex file and returns its path.

        Args:
            tmp_path: Pytest tmp_path fixture.
            content: LaTeX content to write.

        Returns:
            Path to the created .tex file.
        """
        f = tmp_path / "doc.tex"
        f.write_text(content, encoding="utf-8")
        return f

    def test_full_document(self, tmp_path: Path) -> None:
        """Checks end-to-end chunking of a complete LaTeX document."""
        content = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\section{Intro}\n"
            "First paragraph.\n\n"
            "Second paragraph.\n\n"
            "\\section{Methods}\n"
            "Methods paragraph.\n"
            "\\end{document}\n"
        )
        proc = TEXProcessor(self._write_tex(tmp_path, content))
        chunks = proc._obtain_chunks()

        assert len(chunks) == 3
        assert chunks[0]["page"] == "1"
        assert chunks[0]["text"] == "First paragraph."
        assert chunks[1]["page"] == "1"
        assert chunks[1]["text"] == "Second paragraph."
        assert chunks[2]["page"] == "2"
        assert chunks[2]["text"] == "Methods paragraph."
        assert all(c["total_pages"] == 2 for c in chunks)
        assert all(c["doc_type"] == "tex" for c in chunks)

    def test_subsections_end_to_end(self, tmp_path: Path) -> None:
        """Checks that subsection numbering works through the full pipeline."""
        content = (
            "\\begin{document}\n"
            "\\section{Main}\n"
            "Intro text.\n\n"
            "\\subsection{Sub A}\n"
            "Sub A text.\n\n"
            "\\subsection{Sub B}\n"
            "Sub B text.\n"
            "\\end{document}\n"
        )
        proc = TEXProcessor(self._write_tex(tmp_path, content))
        chunks = proc._obtain_chunks()

        assert len(chunks) == 3
        assert [c["page"] for c in chunks] == ["1", "1.1", "1.2"]

    def test_preserves_math(self, tmp_path: Path) -> None:
        """Checks that math expressions survive the full pipeline."""
        content = (
            "\\begin{document}\n"
            "\\section{Math}\n"
            "The formula $E = mc^2$ is important.\n"
            "\\end{document}\n"
        )
        proc = TEXProcessor(self._write_tex(tmp_path, content))
        chunks = proc._obtain_chunks()

        assert "$E = mc^2$" in chunks[0]["text"]

    def test_sets_correct_source(self, tmp_path: Path) -> None:
        """Checks that the source filename is set correctly."""
        content = "\\begin{document}\n\\section{A}\nText.\n\\end{document}\n"
        f = tmp_path / "lecture.tex"
        f.write_text(content, encoding="utf-8")
        proc = TEXProcessor(f)
        chunks = proc._obtain_chunks()

        assert chunks[0]["source"] == "lecture.tex"

    def test_empty_document(self, tmp_path: Path) -> None:
        """Checks that a document with no content returns no chunks."""
        content = "\\begin{document}\n\\end{document}\n"
        proc = TEXProcessor(self._write_tex(tmp_path, content))

        assert proc._obtain_chunks() == []

    def test_skips_empty_paragraphs(self, tmp_path: Path) -> None:
        """Checks that empty paragraphs after stripping are not included."""
        content = (
            "\\begin{document}\n"
            "\\section{A}\n"
            "Real text.\n\n"
            "\\begin{figure}\n\\end{figure}\n\n"
            "More text.\n"
            "\\end{document}\n"
        )
        proc = TEXProcessor(self._write_tex(tmp_path, content))
        chunks = proc._obtain_chunks()

        assert all(c["text"].strip() for c in chunks)
