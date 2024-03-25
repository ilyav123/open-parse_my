from typing import List, Union
from pathlib import Path

import fitz

from src.utils import load_doc
from src import text, tables
from src.schemas import Node, TextElement, TableElement


def elements_to_nodes(elements: List[Union[TextElement, TableElement]]) -> List[Node]:
    raise NotImplementedError


class PipelineStep:
    pass


class DocumentParse:
    def __init__(
        self,
        file: str | Path | fitz.Document,
        processing: List[PipelineStep],
        postprocessing: List[PipelineStep],
        parse_tables: bool = True,
    ):
        doc = load_doc(file)

        text_elems = text.parse(doc)
        if parse_tables:
            table_elems = tables.parse(doc)

        # Parse images (optional)

        # combine elements?

        #

        pass