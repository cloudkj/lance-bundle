import requests
import os
import tempfile
from urllib.parse import urljoin
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from lance_bundle import save, ExportDataset

# PyTorch concurrency
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
# HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}

# Berkshire Hathaway shareholder letter source urls
letter_urls = [
    "https://www.berkshirehathaway.com/letters/1977.html",
    "https://www.berkshirehathaway.com/letters/1978.html",
    "https://www.berkshirehathaway.com/letters/1979.html",
    "https://www.berkshirehathaway.com/letters/1980.html",
    "https://www.berkshirehathaway.com/letters/1981.html",
    "https://www.berkshirehathaway.com/letters/1982.html",
    "https://www.berkshirehathaway.com/letters/1983.html",
    "https://www.berkshirehathaway.com/letters/1984.html",
    "https://www.berkshirehathaway.com/letters/1985.html",
    "https://www.berkshirehathaway.com/letters/1986.html",
    "https://www.berkshirehathaway.com/letters/1987.html",
    "https://www.berkshirehathaway.com/letters/1988.html",
    "https://www.berkshirehathaway.com/letters/1989.html",
    "https://www.berkshirehathaway.com/letters/1990.html",
    "https://www.berkshirehathaway.com/letters/1991.html",
    "https://www.berkshirehathaway.com/letters/1992.html",
    "https://www.berkshirehathaway.com/letters/1993.html",
    "https://www.berkshirehathaway.com/letters/1994.html",
    "https://www.berkshirehathaway.com/letters/1995.html",
    "https://www.berkshirehathaway.com/letters/1996.html",
    "https://www.berkshirehathaway.com/letters/1997.html",
    "https://www.berkshirehathaway.com/letters/1998pdf.pdf",
    "https://www.berkshirehathaway.com/letters/final1999pdf.pdf",
    "https://www.berkshirehathaway.com/letters/2000pdf.pdf",
    "https://www.berkshirehathaway.com/letters/2001pdf.pdf",
    "https://www.berkshirehathaway.com/letters/2002pdf.pdf",
    "https://www.berkshirehathaway.com/letters/2003ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2004ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2005ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2006ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2007ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2008ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2009ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2010ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2011ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2012ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2013ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2014ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2015ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2016ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2017ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2018ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2019ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2020ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2021ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2022ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2023ltr.pdf",
    "https://www.berkshirehathaway.com/letters/2024ltr.pdf",
]
print(f"Found {len(letter_urls)} letters to process.")

# Initialize the embedding model (retaining the handle for later use)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
custom_tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=10000)

# Initialize Docling layout-aware converter and chunker
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
converter = DocumentConverter(
    allowed_formats=[InputFormat.HTML, InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    },
)
chunker = HybridChunker(tokenizer=custom_tokenizer)

# Process, chunk, embed
knowledge_base = []
with tempfile.TemporaryDirectory() as temp_dir:
    for url in tqdm(letter_urls):
        print(f"Processing: {url}")
        try:
            doc_response = requests.get(url, headers=headers)
            doc_response.raise_for_status()
            raw_bytes = doc_response.content

            if raw_bytes.startswith(b"%PDF"):
                is_pdf = True
            else:
                content_type = doc_response.headers.get("Content-Type", "").lower()
                is_pdf = "application/pdf" in content_type or url.split("?")[0].endswith(".pdf")

            file_ext = ".pdf" if is_pdf else ".html"
            temp_file_path = os.path.join(temp_dir, f"temp_letter{file_ext}")

            if is_pdf:
                # Write PDF binary files directly
                with open(temp_file_path, "wb") as f:
                    f.write(doc_response.content)
            else:
                # HTML files are text; fix the encoding and save as clean UTF-8
                clean_html = raw_bytes.decode("windows-1252", errors="replace")
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(clean_html)

            result = converter.convert(temp_file_path)
            chunks = list(chunker.chunk(result.document))
            chunk_texts = [chunk.text for chunk in chunks]
            if not chunk_texts:
                continue

            embeddings = model.encode(chunk_texts).tolist()
            for text, vector in zip(chunk_texts, embeddings):
                knowledge_base.append(
                    {"source": url, "text": text, "embedding": vector}
                )
        except Exception as e:
            print(f"Skipping {url} due to error: {e}")

print(f"Success! Generated {len(knowledge_base)} structure-aware embeddings.")

save(
    model=model,
    dataset=ExportDataset(
        [k["text"] for k in knowledge_base],
        [k["embedding"] for k in knowledge_base],
        metadata=[{"source_url": k["source"]} for k in knowledge_base],
        name="berkshire-hathaway-letters",
        description="Berkshire Hathaway Shareholder Letters 1977-2024",
        source="https://www.berkshirehathaway.com/letters/letters.html",
    ),
    output_path="examples/berkshire_hathaway_letters.zip",
)
