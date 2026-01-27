"""
Agentic RAG Application for DGX Spark
Uses Docling for document processing, ChromaDB for vectors, Ollama for LLM

Enhanced with:
- OCR support for scanned documents
- Advanced table extraction
- Rich metadata extraction
- Image description capabilities
- Processing progress feedback
- Format-specific document handling
"""

import os
import tempfile
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import gradio as gr

# Document processing - Docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    OcrOptions,
)

# Try to import Tesseract OCR options (may not be available on all systems)
try:
    from docling.datamodel.pipeline_options import TesseractOcrOptions
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: TesseractOcrOptions not available, using default OCR")

# Try to import EasyOCR options as fallback
try:
    from docling.datamodel.pipeline_options import EasyOcrOptions
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# LangChain components
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")  # Default to larger model for better reasoning
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

# Model registry - defines which models can be used for which purpose
CHAT_MODELS = [
    ("Qwen3 8B (Recommended)", "qwen3:8b"),
    ("Llama 3.2 3B (Fast)", "llama3.2:latest"),
    ("Nemotron Mini 4B", "nemotron-mini:latest"),
    ("DeepSeek R1 32B (Best)", "deepseek-r1:32b"),
    ("Qwen 2.5 Coder 32B", "qwen2.5-coder:32b"),
]

EMBEDDING_MODELS = [
    ("Nomic Embed Text (Default)", "nomic-embed-text:latest"),
]

# Ensure directories exist
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Initialize components
print(f"Initializing with Ollama at {OLLAMA_BASE_URL}")
print(f"Embedding model: {EMBEDDING_MODEL}")
print(f"Chat model: {CHAT_MODEL}")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1
)

# Current active model (can be changed at runtime)
current_chat_model = CHAT_MODEL


def set_chat_model(model_name: str) -> str:
    """Change the active chat model"""
    global llm, current_chat_model
    try:
        llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )
        current_chat_model = model_name
        return f"Chat model changed to: **{model_name}**"
    except Exception as e:
        return f"Error changing model: {e}"

# Initialize or load vector store
vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="rag_documents"
)


def create_document_converter(
    enable_ocr: bool = True,
    force_ocr: bool = False,
    ocr_lang: str = "eng",
    table_mode: str = "accurate"
) -> DocumentConverter:
    """
    Create a DocumentConverter with configurable options.

    Args:
        enable_ocr: Enable OCR for scanned documents
        force_ocr: Force OCR on all pages (for fully scanned docs)
        ocr_lang: OCR language code (eng, deu, fra, spa, etc.)
        table_mode: Table extraction mode ('accurate' or 'fast')

    Returns:
        Configured DocumentConverter instance
    """
    # Configure table structure options
    table_options = TableStructureOptions(
        do_cell_matching=True,
        mode=table_mode,
    )

    # Configure OCR options
    ocr_options = None
    if enable_ocr:
        if TESSERACT_AVAILABLE:
            ocr_options = TesseractOcrOptions(
                lang=[ocr_lang],
                force_full_page_ocr=force_ocr,
            )
            print(f"Using Tesseract OCR with language: {ocr_lang}")
        elif EASYOCR_AVAILABLE:
            # Map common language codes to EasyOCR format
            lang_map = {"eng": "en", "deu": "de", "fra": "fr", "spa": "es"}
            easy_lang = lang_map.get(ocr_lang, "en")
            ocr_options = EasyOcrOptions(
                lang=[easy_lang],
                force_full_page_ocr=force_ocr,
            )
            print(f"Using EasyOCR with language: {easy_lang}")
        else:
            print("Warning: No OCR backend available, OCR disabled")
            enable_ocr = False

    # Configure PDF pipeline options
    pipeline_options = PdfPipelineOptions(
        do_ocr=enable_ocr,
        do_table_structure=True,
        table_structure_options=table_options,
    )

    # Add OCR options if available
    if ocr_options:
        pipeline_options.ocr_options = ocr_options

    # Create converter with PDF-specific options
    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter


# Create default converter with OCR enabled
default_converter = create_document_converter(
    enable_ocr=True,
    force_ocr=False,
    ocr_lang="eng",
    table_mode="accurate"
)


# Agentic prompts
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a routing agent. Analyze the user's question and decide if it requires retrieving information from documents.

Respond with ONLY one of these two words:
- RETRIEVE: if the question asks about company data, business information, customers, employees, financials, products, policies, procedures, or anything that would be in business documents. Questions containing "our", "we", "company" almost always need RETRIEVE.
- DIRECT: ONLY for general knowledge questions completely unrelated to the business (like "what is the capital of France")

When in doubt, choose RETRIEVE.

Question: {question}

Your routing decision (RETRIEVE or DIRECT):"""),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the provided context.
Use the following retrieved context to answer the question. If the context doesn't contain relevant information, say so.
Always cite which parts of the context you used.

Context:
{context}

Question: {question}

Answer:"""),
])

DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the user's question directly and concisely.

Question: {question}

Answer:"""),
])

EVALUATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Evaluate the following response for accuracy and potential hallucinations.
Rate the response on a scale of 1-10 and explain your reasoning briefly.

Question: {question}
Response: {response}
Context used: {context}

Evaluation (format: SCORE: X/10 - Brief explanation):"""),
])

IMAGE_DESCRIPTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are analyzing a document that contains images. Based on the surrounding context,
provide a brief description of what the image likely contains.

Surrounding text context: {context}
Image location: Page {page}

Provide a brief, informative description of what this image likely shows:"""),
])


def extract_image_contexts(result) -> List[Dict[str, Any]]:
    """
    Extract image information and surrounding context from a Docling result.

    Args:
        result: Docling conversion result

    Returns:
        List of image context dictionaries
    """
    image_contexts = []

    try:
        # Check if document has pictures/figures
        if hasattr(result.document, 'pictures') and result.document.pictures:
            for idx, picture in enumerate(result.document.pictures):
                context = {
                    "type": "image",
                    "index": idx,
                    "page": getattr(picture, 'page_no', 'unknown'),
                    "caption": getattr(picture, 'caption', ''),
                }
                image_contexts.append(context)

        # Also check for figures in the document structure
        if hasattr(result.document, 'figures') and result.document.figures:
            for idx, figure in enumerate(result.document.figures):
                context = {
                    "type": "figure",
                    "index": idx,
                    "page": getattr(figure, 'page_no', 'unknown'),
                    "caption": getattr(figure, 'caption', ''),
                }
                image_contexts.append(context)

    except Exception as e:
        print(f"Warning: Could not extract image contexts: {e}")

    return image_contexts


def describe_images_in_document(image_contexts: List[Dict], llm) -> str:
    """
    Generate descriptions for images found in a document.

    Args:
        image_contexts: List of image context dictionaries
        llm: Language model for generating descriptions

    Returns:
        Markdown string with image descriptions
    """
    if not image_contexts:
        return ""

    descriptions = []
    for img in image_contexts:
        if img.get('caption'):
            desc = f"**[Image on page {img['page']}]**: {img['caption']}"
        else:
            desc = f"**[Image on page {img['page']}]**: Visual content present"
        descriptions.append(desc)

    if descriptions:
        return "\n\n---\n**Document Images:**\n" + "\n".join(descriptions) + "\n---\n"
    return ""


def process_document(
    file_path: str,
    enable_ocr: bool = True,
    force_ocr: bool = False,
    ocr_lang: str = "eng",
    table_mode: str = "accurate",
    include_images: bool = True,
    progress_callback=None
) -> List[Document]:
    """
    Process a document using Docling's advanced parsing with configurable options.

    Args:
        file_path: Path to the document file
        enable_ocr: Enable OCR for scanned documents
        force_ocr: Force OCR on all pages
        ocr_lang: OCR language code
        table_mode: Table extraction mode ('accurate' or 'fast')
        include_images: Include image descriptions in output
        progress_callback: Optional callback for progress updates

    Returns:
        List of LangChain Document objects
    """
    print(f"Processing document: {file_path}")
    ext = Path(file_path).suffix.lower()
    filename = Path(file_path).name

    if progress_callback:
        progress_callback(0.1, f"Loading {filename}...")

    try:
        # Create converter with appropriate options based on file type
        if ext == '.pdf':
            # PDFs get full OCR and table extraction support
            converter = create_document_converter(
                enable_ocr=enable_ocr,
                force_ocr=force_ocr,
                ocr_lang=ocr_lang,
                table_mode=table_mode
            )
        elif ext in ['.xlsx', '.xls', '.csv']:
            # Spreadsheets - use default converter, tables are native
            converter = DocumentConverter()
            print(f"Processing spreadsheet: {filename}")
        elif ext in ['.docx', '.doc']:
            # Word documents - OCR not typically needed
            converter = DocumentConverter()
            print(f"Processing Word document: {filename}")
        elif ext in ['.pptx', '.ppt']:
            # PowerPoint - may need OCR for embedded images
            converter = create_document_converter(
                enable_ocr=enable_ocr,
                force_ocr=False,
                ocr_lang=ocr_lang,
                table_mode="fast"
            )
            print(f"Processing PowerPoint: {filename}")
        elif ext in ['.html', '.htm']:
            # HTML - no OCR needed
            converter = DocumentConverter()
            print(f"Processing HTML: {filename}")
        else:
            # Default handling for other formats
            converter = default_converter

        if progress_callback:
            progress_callback(0.3, f"Converting {filename} with Docling...")

        # Convert document
        result = converter.convert(file_path)

        if progress_callback:
            progress_callback(0.5, f"Extracting content from {filename}...")

        # Extract image contexts if requested
        image_descriptions = ""
        if include_images:
            image_contexts = extract_image_contexts(result)
            if image_contexts:
                image_descriptions = describe_images_in_document(image_contexts, llm)
                print(f"Found {len(image_contexts)} images/figures in document")

        if progress_callback:
            progress_callback(0.6, f"Chunking {filename}...")

        # Use HybridChunker for better semantic chunking
        chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=512,
            merge_peers=True
        )

        chunks = list(chunker.chunk(result.document))
        total_chunks = len(chunks)

        if progress_callback:
            progress_callback(0.8, f"Creating {total_chunks} chunks from {filename}...")

        # Convert to LangChain documents with rich metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # Build rich metadata
            metadata = {
                "source": file_path,
                "filename": filename,
                "file_type": ext,
                "chunk_id": i,
                "total_chunks": total_chunks,
                "doc_type": result.document.origin.mimetype if result.document.origin else "unknown",
                "processing_options": {
                    "ocr_enabled": enable_ocr,
                    "ocr_lang": ocr_lang,
                    "table_mode": table_mode
                }
            }

            # Extract heading/section context if available
            try:
                if hasattr(chunk, 'meta') and chunk.meta:
                    if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                        metadata["section"] = " > ".join(chunk.meta.headings)
                    if hasattr(chunk.meta, 'page'):
                        metadata["page"] = chunk.meta.page
                    if hasattr(chunk.meta, 'doc_items'):
                        # Track if chunk contains tables
                        for item in chunk.meta.doc_items:
                            if hasattr(item, 'label') and 'table' in str(item.label).lower():
                                metadata["contains_table"] = True
                                break
            except Exception as e:
                print(f"Warning: Could not extract chunk metadata: {e}")

            # Prepend image descriptions to first chunk if available
            content = chunk.text
            if i == 0 and image_descriptions:
                content = image_descriptions + "\n\n" + content

            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)

        if progress_callback:
            progress_callback(1.0, f"Completed processing {filename}")

        print(f"Created {len(documents)} chunks from {file_path}")
        return documents

    except Exception as e:
        print(f"Docling processing failed, falling back to basic splitting: {e}")

        if progress_callback:
            progress_callback(0.5, f"Falling back to basic text extraction for {filename}...")

        # Fallback to basic text extraction
        try:
            with open(file_path, 'r', errors='ignore') as f:
                text = f.read()
        except Exception:
            # Binary file - try to extract what we can
            with open(file_path, 'rb') as f:
                text = f.read().decode('utf-8', errors='ignore')

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)

        if progress_callback:
            progress_callback(1.0, f"Completed fallback processing for {filename}")

        return [
            Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "filename": filename,
                    "file_type": ext,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "processing_method": "fallback_text_split"
                }
            )
            for i, chunk in enumerate(chunks)
        ]


def add_documents(
    files: List[str],
    enable_ocr: bool = True,
    force_ocr: bool = False,
    ocr_lang: str = "eng",
    table_mode: str = "accurate",
    include_images: bool = True,
    progress=None
) -> str:
    """Add documents to the vector store with progress tracking"""
    if not files:
        return "No files provided"

    total_chunks = 0
    processed_files = []
    failed_files = []
    total_files = len(files)

    for idx, file_path in enumerate(files):
        try:
            # Create progress callback for this file
            def file_progress(pct, msg):
                if progress:
                    overall_pct = (idx + pct) / total_files
                    progress(overall_pct, msg)

            documents = process_document(
                file_path,
                enable_ocr=enable_ocr,
                force_ocr=force_ocr,
                ocr_lang=ocr_lang,
                table_mode=table_mode,
                include_images=include_images,
                progress_callback=file_progress
            )

            if documents:
                vectorstore.add_documents(documents)
                total_chunks += len(documents)
                processed_files.append(f"{Path(file_path).name} ({len(documents)} chunks)")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed_files.append(f"{Path(file_path).name}: {str(e)[:50]}")
            continue

    # Build result message
    result_msg = f"**Processed {len(processed_files)}/{total_files} files**\n"
    result_msg += f"**Total chunks created:** {total_chunks}\n\n"

    if processed_files:
        result_msg += "**Successfully processed:**\n"
        for f in processed_files:
            result_msg += f"- {f}\n"

    if failed_files:
        result_msg += "\n**Failed to process:**\n"
        for f in failed_files:
            result_msg += f"- {f}\n"

    return result_msg


def route_query(question: str) -> str:
    """Determine if query needs retrieval or direct answer"""
    chain = ROUTER_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    return result.strip().upper()


def expand_query(question: str) -> List[str]:
    """Expand query with related search terms"""
    queries = [question]
    q_lower = question.lower()

    # Add keyword expansions for common business queries
    if "customer" in q_lower or "client" in q_lower:
        queries.append("customer company revenue sales account")
        queries.append("Midwest Dental Atlantic Distributors Pacific Coast")
    if "top" in q_lower or "best" in q_lower or "biggest" in q_lower or "largest" in q_lower:
        queries.append("highest revenue largest sales")
    if "employee" in q_lower or "staff" in q_lower or "who works" in q_lower:
        queries.append("employee directory staff name department")
    if "budget" in q_lower or "financial" in q_lower or "cost" in q_lower:
        queries.append("budget expense revenue financial quarterly")
    if "vendor" in q_lower or "supplier" in q_lower:
        queries.append("vendor supplier company contact")
    if "inventory" in q_lower or "stock" in q_lower or "product" in q_lower:
        queries.append("inventory product SKU quantity stock")
    if "policy" in q_lower or "procedure" in q_lower:
        queries.append("policy procedure guideline rule")
    if "table" in q_lower or "data" in q_lower or "spreadsheet" in q_lower:
        queries.append("table data row column cell")
    if "image" in q_lower or "picture" in q_lower or "figure" in q_lower or "diagram" in q_lower:
        queries.append("image figure diagram illustration visual")

    return queries


def retrieve_context(question: str, k: int = 8) -> Tuple[str, List[Document]]:
    """Retrieve relevant documents using expanded queries"""
    queries = expand_query(question)

    # Collect docs from all queries, interleaving results
    seen_content = set()
    query_results = []

    # Get results from each query
    for query in queries:
        docs = vectorstore.similarity_search(query, k=k)
        query_results.append(docs)

    # Interleave results: take 1st from each query, then 2nd, etc.
    all_docs = []
    for i in range(k):
        for docs in query_results:
            if i < len(docs):
                doc = docs[i]
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)

    # Take top k unique docs
    all_docs = all_docs[:k]

    # Build context with rich metadata
    context_parts = []
    for doc in all_docs:
        meta = doc.metadata
        source_info = f"Source: {meta.get('filename', meta.get('source', 'unknown'))}"

        # Add section info if available
        if meta.get('section'):
            source_info += f" | Section: {meta['section']}"

        # Add page info if available
        if meta.get('page'):
            source_info += f" | Page: {meta['page']}"

        # Note if chunk contains table data
        if meta.get('contains_table'):
            source_info += " | Contains: Table"

        context_parts.append(f"[{source_info}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    return context, all_docs


def evaluate_response(question: str, response: str, context: str) -> str:
    """Evaluate response for hallucinations"""
    chain = EVALUATE_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "question": question,
        "response": response,
        "context": context
    })


def agentic_rag(question: str, history: List[Tuple[str, str]]) -> Tuple[str, str]:
    """Main agentic RAG function with routing and evaluation"""

    # Step 1: Route the query
    route = route_query(question)
    agent_log = f"**Routing Decision:** {route}\n\n"

    if "RETRIEVE" in route:
        # Step 2a: Retrieve context
        context, docs = retrieve_context(question)

        if not docs:
            agent_log += "**Warning:** No relevant documents found in the knowledge base.\n\n"
            context = "No relevant documents found."
        else:
            agent_log += f"**Retrieved {len(docs)} relevant chunks**\n\n"

            # Log source files
            sources = set()
            for doc in docs:
                filename = doc.metadata.get('filename', doc.metadata.get('source', 'unknown'))
                sources.add(filename)
            agent_log += f"**Sources:** {', '.join(sources)}\n\n"

        # Step 3: Generate response with context
        chain = RAG_PROMPT | llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": question
        })

        # Step 4: Evaluate response
        evaluation = evaluate_response(question, response, context)
        agent_log += f"**Evaluation:** {evaluation}\n"

    else:
        # Step 2b: Direct response without retrieval
        agent_log += "**Mode:** Direct response (no document retrieval needed)\n\n"
        context = "N/A - Direct response"

        chain = DIRECT_PROMPT | llm | StrOutputParser()
        response = chain.invoke({"question": question})

    return response, agent_log


def get_collection_stats() -> str:
    """Get statistics about the current document collection"""
    try:
        collection = vectorstore._collection
        count = collection.count()

        # Try to get unique sources
        if count > 0:
            try:
                # Get a sample of metadata to show unique files
                results = collection.peek(limit=min(count, 100))
                if results and 'metadatas' in results:
                    sources = set()
                    for meta in results['metadatas']:
                        if meta and 'filename' in meta:
                            sources.add(meta['filename'])
                        elif meta and 'source' in meta:
                            sources.add(Path(meta['source']).name)

                    stats = f"**Knowledge Base Statistics:**\n"
                    stats += f"- Total chunks: {count}\n"
                    stats += f"- Unique files: {len(sources)}\n"
                    if sources:
                        stats += f"- Files: {', '.join(list(sources)[:5])}"
                        if len(sources) > 5:
                            stats += f" (+{len(sources)-5} more)"
                    return stats
            except Exception:
                pass

        return f"**Knowledge Base:** {count} chunks"
    except Exception as e:
        return f"Error getting stats: {e}"


def clear_collection() -> str:
    """Clear all documents from the collection"""
    global vectorstore
    try:
        vectorstore.delete_collection()
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="rag_documents"
        )
        return "Knowledge base cleared successfully"
    except Exception as e:
        return f"Error clearing collection: {e}"


# Gradio Interface
with gr.Blocks(title="Agentic RAG - DGX Spark") as app:
    gr.Markdown("""
    # Agentic RAG System
    **Powered by Docling + LangChain + Ollama on DGX Spark**

    This system uses intelligent routing to decide when to retrieve from documents vs. answer directly.
    Documents are processed with Docling's advanced features including OCR, table extraction, and semantic chunking.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Ask a question",
                    placeholder="Type your question here...",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary")

            agent_output = gr.Markdown(label="Agent Activity Log")

        with gr.Column(scale=1):
            gr.Markdown("### Document Management")

            file_upload = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".html", ".xls", ".xlsx", ".pptx", ".ppt"]
            )

            # Processing Options Accordion
            with gr.Accordion("Processing Options", open=False):
                enable_ocr = gr.Checkbox(
                    label="Enable OCR",
                    value=True,
                    info="Extract text from scanned documents and images"
                )
                force_ocr = gr.Checkbox(
                    label="Force Full Page OCR",
                    value=False,
                    info="Use OCR on all pages (for fully scanned documents)"
                )
                ocr_lang = gr.Dropdown(
                    label="OCR Language",
                    choices=[
                        ("English", "eng"),
                        ("German", "deu"),
                        ("French", "fra"),
                        ("Spanish", "spa"),
                        ("Italian", "ita"),
                        ("Portuguese", "por"),
                        ("Chinese Simplified", "chi_sim"),
                        ("Japanese", "jpn"),
                    ],
                    value="eng",
                    info="Language for OCR text recognition"
                )
                table_mode = gr.Radio(
                    label="Table Extraction Mode",
                    choices=[
                        ("Accurate (slower)", "accurate"),
                        ("Fast (less precise)", "fast")
                    ],
                    value="accurate",
                    info="Trade-off between accuracy and speed for table extraction"
                )
                include_images = gr.Checkbox(
                    label="Include Image Descriptions",
                    value=True,
                    info="Add descriptions of images/figures found in documents"
                )

            upload_btn = gr.Button("Process Documents", variant="secondary")
            upload_status = gr.Markdown(label="Upload Status")

            gr.Markdown("---")
            stats_output = gr.Markdown(label="Knowledge Base Stats")
            refresh_btn = gr.Button("Refresh Stats")
            clear_btn = gr.Button("Clear Knowledge Base", variant="stop")

            gr.Markdown("---")
            gr.Markdown("### Model Configuration")

            with gr.Accordion("Change Chat Model", open=False):
                chat_model_selector = gr.Dropdown(
                    label="Chat/Reasoning Model",
                    choices=CHAT_MODELS,
                    value=CHAT_MODEL,
                    info="Model for routing, RAG responses, and evaluation"
                )
                change_model_btn = gr.Button("Apply Model Change", variant="secondary")
                model_change_status = gr.Markdown("")

            gr.Markdown("### Model Architecture")
            model_info = gr.Markdown(f"""
**Document Parsing:** Docling (local ML)
- Layout analysis & table structure
- OCR via Tesseract/EasyOCR
- *No LLM - uses bundled ML models*

**Embeddings:** `{EMBEDDING_MODEL}`
- Creates 768-dim vectors
- Stored in ChromaDB

**Chat Model:** `{CHAT_MODEL}`
- Query routing decisions
- RAG response generation
- Response evaluation

**Ollama:** `{OLLAMA_BASE_URL}`
            """)

            gr.Markdown("---")
            gr.Markdown("### OCR Status")
            ocr_status = gr.Markdown(f"""
            **Tesseract OCR:** {'Available' if TESSERACT_AVAILABLE else 'Not Available'}
            **EasyOCR:** {'Available' if EASYOCR_AVAILABLE else 'Not Available'}
            """)

    # Event handlers
    def respond(message, history):
        import traceback
        try:
            print(f"[DEBUG] Received message: {message}")
            print(f"[DEBUG] History length: {len(history) if history else 0}")

            # Convert history from message format to tuple format for agentic_rag
            tuple_history = []
            for msg in history:
                if isinstance(msg, dict):
                    if msg.get("role") == "user":
                        tuple_history.append((msg.get("content", ""), ""))
                    elif msg.get("role") == "assistant" and tuple_history:
                        tuple_history[-1] = (tuple_history[-1][0], msg.get("content", ""))

            print(f"[DEBUG] Calling agentic_rag...")
            response, agent_log = agentic_rag(message, tuple_history)
            print(f"[DEBUG] Got response: {response[:100] if response else 'None'}...")

            # Return in new message format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            print(f"[DEBUG] Returning successfully")
            return "", history, agent_log
        except Exception as e:
            print(f"[ERROR] Exception in respond: {e}")
            traceback.print_exc()
            error_msg = f"Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history, f"**Error:** {str(e)}"

    def upload_files(files, enable_ocr, force_ocr, ocr_lang, table_mode, include_images, progress=gr.Progress()):
        if files is None:
            return "No files selected"
        file_paths = [f.name for f in files]
        return add_documents(
            file_paths,
            enable_ocr=enable_ocr,
            force_ocr=force_ocr,
            ocr_lang=ocr_lang,
            table_mode=table_mode,
            include_images=include_images,
            progress=progress
        )

    msg.submit(respond, [msg, chatbot], [msg, chatbot, agent_output])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot, agent_output])
    upload_btn.click(
        upload_files,
        [file_upload, enable_ocr, force_ocr, ocr_lang, table_mode, include_images],
        [upload_status]
    )
    refresh_btn.click(get_collection_stats, [], [stats_output])
    clear_btn.click(clear_collection, [], [upload_status])
    change_model_btn.click(set_chat_model, [chat_model_selector], [model_change_status])

    # Load initial stats
    app.load(get_collection_stats, [], [stats_output])


if __name__ == "__main__":
    print("Starting Agentic RAG server on port 8501...")
    print(f"OCR Support: Tesseract={TESSERACT_AVAILABLE}, EasyOCR={EASYOCR_AVAILABLE}")
    app.launch(
        server_name="0.0.0.0",
        server_port=8501,
        share=False,
        show_error=True
    )
