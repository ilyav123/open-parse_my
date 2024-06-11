from typing import List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from openparse.pdf import Pdf
from openparse.schemas import Bbox, TableElement
from openparse.tables.utils import crop_img_with_padding, adjust_bbox_with_padding
from openparse.tables.table_transformers.schemas import _TableModelOutput
from openparse.tables.utils import convert_img_cords_to_pdf_cords
import os
from datetime import datetime
from datetime import datetime
from PIL import Image, ImageDraw

from .unitable.utils import (
    build_table_from_html_and_cell,  # cell-content-detection
)
from .utils import (
    convert_croppped_cords_to_full_img_cords,  # cell-content-detection
    convert_img_cords_to_pdf_cords,  # cell-content-detection
)
import fitz
import re

from . import pymupdf


class ParsingArgs(BaseModel):
    parsing_algorithm: str
    table_output_format: Literal["str", "markdown", "html"] = Field(default="html")


class TableTransformersArgs(BaseModel):
    parsing_algorithm: Literal["table-transformers"] = Field(
        default="table-transformers"
    )
    min_table_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    min_cell_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    table_output_format: Literal["str", "markdown", "html"] = Field(default="html")

    model_config = ConfigDict(extra="forbid")


class PyMuPDFArgs(BaseModel):
    parsing_algorithm: Literal["pymupdf"] = Field(default="pymupdf")
    table_output_format: Literal["str", "markdown", "html"] = Field(default="html")

    model_config = ConfigDict(extra="forbid")


class UnitableArgs(BaseModel):
    parsing_algorithm: Literal["unitable"] = Field(default="unitable")
    min_table_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    table_output_format: Literal["html"] = Field(default="html")
    table_detection_model: Literal["YOLOX","MS"]=Field(default="MS")
    padding_pct:float=Field(default=0.05,ge=0.0,le=0.1)
    model_config = ConfigDict(extra="forbid")


def _ingest_with_pymupdf(
    doc: Pdf,
    parsing_args: PyMuPDFArgs,
    verbose: bool = False,
) -> List[TableElement]:
    pdoc = doc.to_pymupdf_doc()
    tables = []
    for page_num, page in enumerate(pdoc):
        tabs = page.find_tables()
        for i, tab in enumerate(tabs.tables):
            headers = tab.header.names
            for j, header in enumerate(headers):
                if header is None:
                    headers[j] = ""
                else:
                    headers[j] = header.strip()
            lines = tab.extract()

            if parsing_args.table_output_format == "str":
                text = pymupdf.output_to_markdown(headers, lines)
            elif parsing_args.table_output_format == "markdown":
                text = pymupdf.output_to_markdown(headers, lines)
            elif parsing_args.table_output_format == "html":
                text = pymupdf.output_to_html(headers, lines)

            if verbose:
                print(f"Page {page_num} - Table {i + 1}:\n{text}\n")

            # Flip y-coordinates to match the top-left origin system
            bbox = pymupdf.combine_header_and_table_bboxes(tab.bbox, tab.header.bbox)
            fy0 = page.rect.height - bbox[3]
            fy1 = page.rect.height - bbox[1]

            table = TableElement(
                bbox=Bbox(
                    page=page_num,
                    x0=bbox[0],
                    y0=fy0,
                    x1=bbox[2],
                    y1=fy1,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                ),
                text=text,
            )
            tables.append(table)
    return tables


def _ingest_with_table_transformers(
    doc: Pdf,
    args: TableTransformersArgs,
    verbose: bool = False,
) -> List[TableElement]:
    try:
        from openparse.tables.utils import doc_to_imgs

        from .table_transformers.ml import find_table_bboxes, get_table_content
    except ImportError as e:
        raise ImportError(
            "Table detection and extraction requires the `torch`, `torchvision` and `transformers` libraries to be installed.",
            e,
        )
    pdoc = doc.to_pymupdf_doc()  # type: ignore
    pdf_as_imgs = doc_to_imgs(pdoc)

    pages_with_tables = {}
    for page_num, img in enumerate(pdf_as_imgs):
        pages_with_tables[page_num] = find_table_bboxes(img, args.min_table_confidence)

    tables = []
    for page_num, table_bboxes in pages_with_tables.items():
        page = pdoc[page_num]
        page_dims = (page.rect.width, page.rect.height)
        for table_bbox in table_bboxes:
            table = get_table_content(
                page_dims,
                pdf_as_imgs[page_num],
                table_bbox.bbox,
                args.min_cell_confidence,
                verbose,
            )
            table._run_ocr(page)

            if args.table_output_format == "str":
                table_text = table.to_str()
            elif args.table_output_format == "markdown":
                table_text = table.to_markdown_str()
            elif args.table_output_format == "html":
                table_text = table.to_html_str()

            # Flip y-coordinates to match the top-left origin system
            # FIXME: incorporate padding into bbox
            fy0 = page.rect.height - table_bbox.bbox[3]
            fy1 = page.rect.height - table_bbox.bbox[1]

            table_elem = TableElement(
                bbox=Bbox(
                    page=page_num,
                    x0=table_bbox.bbox[0],
                    y0=fy0,
                    x1=table_bbox.bbox[2],
                    y1=fy1,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                ),
                text=table_text,
            )
            if verbose:
                print(f"Page {page_num}:\n{table_text}\n")

            tables.append(table_elem)

    return tables



# def output_table_images(pdoc,pdf_as_imgs,table_dict,file_prefix):
#     import os
#     from datetime import datetime
#     filename = os.path.basename(doc.file_path)
#     current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
#     for page_num,table_bboxes in table_dict.items():
#         page = pdoc[page_num]
#         for table_bbox in table_bboxes:
#             padded_bbox = adjust_bbox_with_padding(
#             bbox=table_bbox.bbox,
#             page_width=page.rect.width,
#             page_height=page.rect.height,
#             padding_pct=padding_pct,
#             )
#             table_img = crop_img_with_padding(pdf_as_imgs[page_num], padded_bbox)
#             new_filename = f"{filename}_{current_datetime}_{file_prefix}_p{page_num}-T{table_bboxes.index(table_bbox)}.png"
#             file_path = os.path.join('debug_imgs', new_filename)
#             table_img.save(file_path, 'PNG')
#             print(f"Image saved to {file_path}")

def log_table_image(pdf_filename,page_num,table_num,img,comment_str,bboxes=None):

    table_img_copy=img.copy()
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_filename = f"{pdf_filename}_{current_datetime}_P{page_num}_T{table_num}_{comment_str}"
    if bboxes:
        image_filename+=f"_with_rect"
        draw = ImageDraw.Draw(table_img_copy)
        for bbox in bboxes:
            draw.rectangle(bbox, outline="red", width=1)
    image_filename+=".png"
    file_path = os.path.join('debug_imgs', image_filename)
    table_img_copy.save(file_path, 'PNG')
    print(f"Image with predicted bboxes saved to {file_path}")

def log_page_image(doc,page_num,comment_str,table_bboxes=None,current_page_predicted_cells=None,scale=2):
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_filename = os.path.basename(doc.file_path)
    image_filename = f"{pdf_filename}_{current_datetime}_P{page_num}_{comment_str}"
    pdoc = doc.to_pymupdf_doc()
    page=pdoc[page_num]
    words = page.get_text("words")
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    for word in words:
        x0, y0, x1, y1, word_text = word[:5]
        scaled_coords=convert_img_cords_to_pdf_cords((x0,y0,x1,y1),img.size,(page.rect.width, page.rect.height))
        draw.rectangle(scaled_coords, outline="blue", width=1)
    if current_page_predicted_cells:
        image_filename+="_pred_cells"
        for table_cells in current_page_predicted_cells:
            for pred_cell in table_cells:
                scaled_coords=convert_img_cords_to_pdf_cords(pred_cell,img.size,(page.rect.width, page.rect.height))
                draw.rectangle(scaled_coords, outline="red", width=1)
    if table_bboxes:
        for table_bbox in table_bboxes:
            scaled_coords=convert_img_cords_to_pdf_cords(table_bbox,img.size,(page.rect.width, page.rect.height))
            draw.rectangle(scaled_coords, outline="green", width=3)
    image_filename+='.png'
    file_path = os.path.join('debug_imgs', image_filename)
    img.save(file_path,'PNG')
    print(f"Page Image saved as {file_path}")



def log_table_to_file(text, filename,doc,page_num,table_num,comment_str):
    current_pdf_filename = os.path.basename(doc.file_path)
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    s = f"PDF FILE: {current_pdf_filename} TIME: {current_datetime} Page: {page_num} Table: {table_num} Comment: {comment_str}"
    with open(filename, 'a') as file:
        file.write(f"{s}<BR>{text}")


def make_text(words):
    """Return textstring output of get_text("words").

    Word items are sorted for reading sequence left to right,
    top to bottom.
    """
    line_dict = {}  # key: vertical coordinate, value: list of words
    words.sort(key=lambda w: w[0])  # sort by horizontal coordinate
    for w in words:  # fill the line dictionary
        y1 = round(w[3], 1)  # bottom of a word: don't be too picky!
        word = w[4]  # the text of the word
        line = line_dict.get(y1, [])  # read current line content
        line.append(word)  # append new word
        line_dict[y1] = line  # write back to dict
    lines = list(line_dict.items())
    lines.sort()  # sort vertically
    return "\n".join([" ".join(line[1]) for line in lines])

def find_intersecting_words(page,cell_rect,intersection_level):
    words = page.get_text("words")
    cell_rect_area=cell_rect.get_area()
    output_words=[]
    for word in words:
        intersection_area=0
        word_rect=fitz.Rect(word[:4])
        intersection_rect = word_rect & cell_rect
        if intersection_rect.is_empty:
            continue
        else:
            intersection_area = intersection_rect.get_area()
        word_rect_area=word_rect.get_area()
        if word_rect_area==0:
            print("WARNING word rect are=0")
            continue
        ratio=intersection_area/word_rect_area
        if ratio>=intersection_level:
            output_words.append(word)
    return make_text(output_words)


def _ingest_with_unitable(
    doc: Pdf,
    args: UnitableArgs,
    verbose: bool = False,
) -> List[TableElement]:
    try:
        from openparse.tables.utils import doc_to_imgs
        from .table_transformers.ml import find_table_bboxes
        from .unitable.core import table_img_to_html

    except ImportError as e:
        raise ImportError(
            "Table detection and extraction requires the `torch`, `torchvision` and `transformers` libraries to be installed.",
            e,
        )
    pdoc = doc.to_pymupdf_doc()  # type: ignore
    pdf_as_imgs = doc_to_imgs(pdoc)

    pages_with_tables = {}
    padding_pct = args.padding_pct
    table_detection_model=args.table_detection_model


    if table_detection_model=='MS':
        for page_num, img in enumerate(pdf_as_imgs):
            pages_with_tables[page_num] = find_table_bboxes(img, args.min_table_confidence)

    #my code!
    if table_detection_model=='YOLOX':
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(filename=doc.file_path, content_type="application/pdf", languages=["rus","eng"], strategy="hi_res", infer_table_structure=True)
        for el in elements:
            if el.category!="Table":
                continue
            el_bbox=(el.metadata.coordinates.points[0][0],el.metadata.coordinates.points[0][1],el.metadata.coordinates.points[2][0],el.metadata.coordinates.points[2][1])
            el_page_number=el.metadata.page_number-1
            page_size=pdf_as_imgs[el_page_number].size
            el_img_size=(el.metadata.coordinates.system.width,el.metadata.coordinates.system.height)
            bbox_new=convert_img_cords_to_pdf_cords(el_bbox,page_size,el_img_size)
            t_object=_TableModelOutput(label='table',confidence=el.metadata.detection_class_prob,bbox=bbox_new)
            if el_page_number not in pages_with_tables:
                pages_with_tables[el_page_number] = []
            pages_with_tables[el_page_number].append(t_object)
        

    #output_table_images(pdoc,pdf_as_imgs,pages_with_tables,table_detection_model)
  
    
    # my code end


    tables = []
    for page_num, table_bboxes in pages_with_tables.items():
        page = pdoc[page_num]
        current_page_predicted_cells=[] #list of lists where each internal list is a list of cells predicted for the table in the current page
        current_page_table_bboxes=[]
        for table_bbox in table_bboxes:
            #padding_pct = 0.05
            padded_bbox = adjust_bbox_with_padding(
                bbox=table_bbox.bbox,
                page_width=page.rect.width,
                page_height=page.rect.height,
                padding_pct=padding_pct,
            )
            current_page_table_bboxes.append(padded_bbox)
            table_img = crop_img_with_padding(pdf_as_imgs[page_num], padded_bbox)
            log_table_image(os.path.basename(doc.file_path),page_num,table_bboxes.index(table_bbox),table_img,f"{table_detection_model}_padding{int(padding_pct*100)}")
            

            (table_str,pred_html,pred_bbox) = table_img_to_html(table_img)
            log_table_image(os.path.basename(doc.file_path),page_num,table_bboxes.index(table_bbox),table_img,f"{table_detection_model}_padding{int(padding_pct*100)}",pred_bbox)

            pred_bbox_pdf_coords=[]
            for bbox in pred_bbox:
                img_coord_bbox=convert_croppped_cords_to_full_img_cords(0,table_img.size,bbox,padded_bbox)
                pdf_coord_bbox=convert_img_cords_to_pdf_cords(img_coord_bbox,(page.rect.width, page.rect.height),pdf_as_imgs[page_num].size)
                pred_bbox_pdf_coords.append(pdf_coord_bbox)
            current_page_predicted_cells.append(pred_bbox_pdf_coords)
            #log_pdf_page_image(doc,page_num,)

            # my code start of my block
            pred_cell_lst=[]
            for bbox in pred_bbox_pdf_coords:
                cell_rect = fitz.Rect(bbox)
                text_simple = page.get_textbox(cell_rect)
                text=find_intersecting_words(page,cell_rect,0.5)
                #text = re.sub(r'\n\n', r'__DOUBLE_NEWLINE__', text)
                #text = re.sub(r'\n', '', text)
                #text = re.sub(r'__DOUBLE_NEWLINE__', r'\n', text)
                pred_cell_lst.append(text)


            table_str_lst = build_table_from_html_and_cell(pred_html[1:], pred_cell_lst)
            table_str = "".join(table_str_lst)
            table_str = '<style>table, th, td {border: 1px solid black;font-size: 10px;}</style><table>' + table_str + "</table>"

           
            log_table_to_file(table_str+"<BR><BR>", "table_str.html",doc,page_num,table_bboxes.index(table_bbox),f"{table_detection_model}")
            # end of my block


            # Flip y-coordinates to match the top-left origin system
            fy0 = page.rect.height - padded_bbox[3]
            fy1 = page.rect.height - padded_bbox[1]

            table_elem = TableElement(
                bbox=Bbox(
                    page=page_num,
                    x0=padded_bbox[0],
                    y0=fy0,
                    x1=padded_bbox[2],
                    y1=fy1,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                ),
                text=table_str,
            )

            tables.append(table_elem)
        log_page_image(doc,page_num,f"FULLPAGE_{table_detection_model}_padding{int(padding_pct*100)}",current_page_table_bboxes,current_page_predicted_cells,scale=2)

    return tables


def ingest(
    doc: Pdf,
    parsing_args: Union[TableTransformersArgs, PyMuPDFArgs, UnitableArgs, None] = None,
    verbose: bool = False,
) -> List[TableElement]:
    if isinstance(parsing_args, TableTransformersArgs):
        return _ingest_with_table_transformers(doc, parsing_args, verbose)
    elif isinstance(parsing_args, PyMuPDFArgs):
        return _ingest_with_pymupdf(doc, parsing_args, verbose)
    elif isinstance(parsing_args, UnitableArgs):
        return _ingest_with_unitable(doc, parsing_args, verbose)
    else:
        raise ValueError("Unsupported parsing_algorithm.")
