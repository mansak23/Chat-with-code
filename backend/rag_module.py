import os
import json
import uuid
import re
import tempfile
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

import tree_sitter
from tree_sitter_languages import get_language

load_dotenv()

# Load model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_FUNCTION_CHROMA = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')

DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 1.3

COLLECTION_NAME = "code_chunks"

client = None
collection = None

# --- Tree-sitter Language and Parser Setup ---
C_LANGUAGE = None
CPP_LANGUAGE = None
C_PARSER = None
CPP_PARSER = None

try:
    C_LANGUAGE = get_language('c')
    C_PARSER = tree_sitter.Parser()
    C_PARSER.set_language(C_LANGUAGE)

    CPP_LANGUAGE = get_language('cpp')
    CPP_PARSER = tree_sitter.Parser()
    CPP_PARSER.set_language(CPP_LANGUAGE)

except Exception as e:
    print(f"Error loading tree-sitter C/C++ parsers: {e}")
    print("Please ensure 'tree_sitter_languages' can find and load the 'c' and 'cpp' grammars.")
    print("You might need to install 'tree_sitter' and 'tree_sitter_languages' using pip.")
    exit("Cannot proceed without tree-sitter parsers.")


# --- Tree-sitter Query Patterns for C++ ---
CPP_FUNCTION_QUERY_PATTERN = r"""
    (function_definition
        declarator: (function_declarator) @function.declarator_node
        body: (compound_statement) @function.body
    ) @function.definition
"""

CPP_CLASS_QUERY_PATTERN = r"""
    (class_specifier
        name: (type_identifier) @class.name
        body: (field_declaration_list) @class.body
    ) @class.definition
"""

CPP_STRUCT_QUERY_PATTERN = r"""
    (struct_specifier
        name: (type_identifier) @struct.name
        body: (field_declaration_list) @struct.body
    ) @struct.definition
"""

CPP_GLOBAL_ARRAY_INIT_QUERY_PATTERN = r"""
    (init_declarator
        declarator: (array_declarator
            (identifier) @array.name
            "[" "]"
        )
        value: (initializer_list) @array.body
    ) @array.definition
    (declaration
        (array_declarator
            (identifier) @array.name
            "[" "]"
        )
        (initializer_list) @array.body
    ) @array.definition
"""

# --- Tree-sitter Query Patterns for C (ADJUSTED for 'Invalid node type type_declarator') ---
C_FUNCTION_QUERY_PATTERN = r"""
    (function_definition
        declarator: (function_declarator) @function.declarator_node
        body: (compound_statement) @function.body
    ) @function.definition
"""

C_CLASS_QUERY_PATTERN = r""

# REVISED C_STRUCT_QUERY_PATTERN to avoid 'type_declarator' in query
# Instead, capture the 'type_identifier' directly as the alias.
C_STRUCT_QUERY_PATTERN = r"""
    (struct_specifier
        name: (type_identifier) @struct.name
        body: (field_declaration_list) @struct.body
    ) @struct.definition

    (type_definition ; for typedef struct { ... } alias; in C
        type: (struct_specifier
            name: (type_identifier) @typedef_struct_name ; for optional explicit struct name
            body: (field_declaration_list) @typedef_struct_body
        )
        ; Capture the type_identifier directly at the end of typedef_definition
        (type_identifier) @typedef_alias_name ; Capture the alias name directly
    ) @typedef_struct.definition
"""

C_GLOBAL_ARRAY_INIT_QUERY_PATTERN = r"""
    (declaration
        declarator: (init_declarator
            declarator: (pointer_declarator
                (array_declarator
                    declarator: (identifier) @array.name
                )
            )
            value: (initializer_list)
        )
    ) @array.definition
    (declaration
        declarator: (init_declarator
            declarator: (array_declarator
                declarator: (identifier) @array.name
                size: (number_literal)
            )
            value: (initializer_list)
        )
    ) @array.definition
"""


# Universal Comment Query
COMMENT_QUERY_PATTERN = r"""
    (comment) @comment.block
"""

TEST_FUNCTION_NAME_PATTERN = re.compile(
    r'(?:^|_)test_[\w\d_]*$|'
    r'^[A-Z_]+_TEST_[\w\d_]*$|'
    r'^[A-Z_]+_CASE_[\w\d_]*$|'
    r'Test[A-Z][\w\d_]*$'
, re.IGNORECASE)


# === Code Chunk Extraction (using Tree-sitter) ===
def extract_code_chunks(file_path: str) -> List[Dict[str, Any]]:
    chunks = []
    
    file_extension = os.path.splitext(file_path)[1].lower()

    current_parser = None
    current_language = None
    
    if file_extension in ['.cpp', '.hpp', '.cxx']:
        current_parser = CPP_PARSER
        current_language = CPP_LANGUAGE
        func_pattern_str = CPP_FUNCTION_QUERY_PATTERN
        class_pattern_str = CPP_CLASS_QUERY_PATTERN
        struct_pattern_str = CPP_STRUCT_QUERY_PATTERN
        array_pattern_str = CPP_GLOBAL_ARRAY_INIT_QUERY_PATTERN
    elif file_extension in ['.c', '.h']:
        current_parser = C_PARSER
        current_language = C_LANGUAGE
        func_pattern_str = C_FUNCTION_QUERY_PATTERN
        class_pattern_str = C_CLASS_QUERY_PATTERN
        struct_pattern_str = C_STRUCT_QUERY_PATTERN
        array_pattern_str = C_GLOBAL_ARRAY_INIT_QUERY_PATTERN
        
        if C_PARSER is None:
             print(f"Warning: C parser not loaded, falling back to C++ parser for {file_path}")
             current_parser = CPP_PARSER
             current_language = CPP_LANGUAGE
             func_pattern_str = CPP_FUNCTION_QUERY_PATTERN
             class_pattern_str = CPP_CLASS_QUERY_PATTERN
             struct_pattern_str = CPP_STRUCT_QUERY_PATTERN
             array_pattern_str = CPP_GLOBAL_ARRAY_INIT_QUERY_PATTERN
    else:
        print(f"Warning: Unsupported file extension '{file_extension}'. Skipping {file_path}")
        return []

    if current_parser is None or current_language is None:
        print(f"Error: No suitable parser found for file {file_path}")
        return []

    try:
        function_query = current_language.query(func_pattern_str)
        class_query = current_language.query(class_pattern_str) if class_pattern_str else None
        struct_query = current_language.query(struct_pattern_str)
        global_array_init_query = current_language.query(array_pattern_str)
        comment_query = current_language.query(COMMENT_QUERY_PATTERN)
    except Exception as e:
        print(f"Error compiling tree-sitter queries for {file_extension} files: {e}")
        print("--- Failed Query Patterns Debug ---")
        print(f"Function Query:\n{func_pattern_str}")
        if class_pattern_str: print(f"Class Query:\n{class_pattern_str}")
        print(f"Struct Query:\n{struct_pattern_str}")
        print(f"Array Query:\n{array_pattern_str}")
        print(f"Error details: {e}")
        print("This often means the query syntax does not match the specific grammar version or language.")
        return []

    try:
        with open(file_path, 'rb') as f:
            code_bytes = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    tree = current_parser.parse(code_bytes)
    root_node = tree.root_node

    # --- Helper Functions (defined within extract_code_chunks scope) ---
    def get_node_content(node):
        return code_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')

    def get_name_from_capture(captures, name_suffix):
        for node, name in captures:
            if name.endswith(name_suffix):
                return node.text.decode('utf-8', errors='ignore')
        return ""

    def get_function_name_and_scope(func_declarator_node):
        func_name = ""
        scope_name = ""

        name_node = None
        for child in func_declarator_node.children:
            if child.type == 'qualified_identifier':
                name_node = child
                break
            elif child.type == 'identifier' or child.type == 'field_identifier':
                name_node = child
                
        if name_node:
            func_name_raw = name_node.text.decode('utf-8', errors='ignore')
            
            if name_node.type == 'qualified_identifier':
                parts = func_name_raw.split('::')
                if len(parts) > 1:
                    scope_name = "::".join(parts[:-1])
                    func_name = parts[-1]
                else:
                    func_name = func_name_raw
            else:
                func_name = func_name_raw

        func_name = func_name.split('(')[0].strip()

        final_func_name = func_name
        if scope_name and func_name and not func_name.startswith(f"{scope_name}::"):
            final_func_name = f"{scope_name}::{func_name}"
            
        return final_func_name, scope_name


    processed_nodes = set()
    scope_stack = []
    cursor = root_node.walk()

    def add_chunk(node, chunk_type, func_name="", class_name="", struct_name="", array_name="", test_case_function_name=""):
        node_id = (node.start_byte, node.end_byte, node.type)

        for existing_node_id in processed_nodes:
            existing_start, existing_end, _ = existing_node_id
            if (existing_start <= node.start_byte and existing_end >= node.end_byte) or \
               (node.start_byte <= existing_start and node.end_byte >= existing_end):
                return

        content = get_node_content(node)
        
        current_class_name_in_scope = ""
        current_struct_name_in_scope = ""
        for scope_type, scope_name, scope_end_byte in reversed(scope_stack):
            if node.start_byte < scope_end_byte:
                if scope_type == "class":
                    current_class_name_in_scope = scope_name
                    break
                elif scope_type == "struct":
                    current_struct_name_in_scope = scope_name
                    break
        
        final_class_name = class_name if class_name else current_class_name_in_scope
        final_struct_name = struct_name if struct_name else current_struct_name_in_scope

        chunks.append({
            "content": content,
            "source": os.path.basename(file_path),
            "start_line": node.start_point[0] + 1,
            "type": chunk_type,
            "function_name": func_name,
            "class_name": final_class_name,
            "struct_name": final_struct_name,
            "array_name": array_name,
            "test_case_function_name": test_case_function_name
        })
        processed_nodes.add(node_id)


    # --- AST Traversal Logic ---
    if not cursor.goto_first_child():
        for capture_node, capture_name in comment_query.captures(root_node):
            if capture_name == "comment.block":
                add_chunk(capture_node, "comment")
        return chunks

    while True:
        node = cursor.node

        while scope_stack and node.start_byte >= scope_stack[-1][2]:
            scope_stack.pop()

        # 1. Class Definitions (Only for C++ files)
        if node.type == 'class_specifier' and class_query:
            captures = class_query.captures(node)
            class_name = get_name_from_capture(captures, "class.name")
            if class_name:
                add_chunk(node, "class", class_name=class_name)
                scope_stack.append(("class", class_name, node.end_byte))
        
        # 2. Struct Definitions (C and C++)
        elif node.type == 'struct_specifier' and struct_query:
            captures = struct_query.captures(node)
            struct_name = get_name_from_capture(captures, "struct.name")
            if struct_name:
                add_chunk(node, "struct", struct_name=struct_name)
                scope_stack.append(("struct", struct_name, node.end_byte))

        elif node.type == 'type_definition' and struct_query:
            captures = struct_query.captures(node)
            typedef_struct_name = get_name_from_capture(captures, "typedef_struct_name")
            typedef_alias_name = get_name_from_capture(captures, "typedef_alias_name")

            if typedef_alias_name:
                add_chunk(node, "struct", struct_name=typedef_alias_name)
                scope_stack.append(("struct", typedef_alias_name, node.end_byte))
            elif typedef_struct_name:
                add_chunk(node, "struct", struct_name=typedef_struct_name)
                scope_stack.append(("struct", typedef_struct_name, node.end_byte))


        # 3. Function Definitions (global and member)
        elif node.type == 'function_definition' and function_query:
            func_definition_captures = function_query.captures(node)
            
            func_declarator_node = None
            for n, name in func_definition_captures:
                if name == "function.declarator_node":
                    func_declarator_node = n
                    break

            if func_declarator_node:
                func_name, scope_name_from_func = get_function_name_and_scope(func_declarator_node)
                
                chunk_type = "function"
                test_case_function_name = ""
                if TEST_FUNCTION_NAME_PATTERN.search(func_name):
                    chunk_type = "test_case_function"
                    test_case_function_name = func_name
                
                add_chunk(node, chunk_type, func_name=func_name, test_case_function_name=test_case_function_name,
                          class_name=scope_name_from_func if scope_name_from_func and current_language == CPP_LANGUAGE else "",
                          struct_name=scope_name_from_func if scope_name_from_func and current_language == C_LANGUAGE else ""
                          )

        # 4. Global Array Initializations
        elif node.type == 'declaration' and global_array_init_query:
            captures = global_array_init_query.captures(node)
            array_name = get_name_from_capture(captures, "array.name")
            if array_name and any(n for n,n_name in captures if n_name.endswith("array.definition")):
                add_chunk(node, "array_init", array_name=array_name)

        # 5. Comments
        elif node.type == 'comment' and comment_query:
            add_chunk(node, "comment")


        # --- Traversal Control ---
        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        
        while True:
            if not cursor.goto_parent():
                return chunks
            if cursor.goto_next_sibling():
                break
    
    chunks.sort(key=lambda x: x['start_line'])
    return chunks


# === File Processing ===
def process_and_store_uploaded_file(file_content_bytes: bytes, original_filename: str):
    """
    Processes a single uploaded file's content and stores its chunks in ChromaDB.
    """
    global collection
    if collection is None:
        print("Error: ChromaDB collection not initialized. Cannot process uploaded file.")
        raise RuntimeError("ChromaDB collection not initialized.")

    suffix = os.path.splitext(original_filename)[1].lower()
    if suffix not in ['.c', '.cpp', '.h', '.hpp', '.cxx']:
        suffix = '.cpp'

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=suffix) as temp_f:
        temp_f.write(file_content_bytes)
        temp_file_path = temp_f.name

    try:
        print(f"Processing uploaded file: {original_filename} (temp path: {temp_file_path})...")
        chunks = extract_code_chunks(temp_file_path)

        ids_batch = []
        embeddings_batch = []
        documents_batch = []
        metadatas_batch = []
        batch_size = 100

        for chunk in chunks:
            chunk_unique_id_str = f"{original_filename}-{chunk['start_line']}-{hash(chunk['content'])}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_unique_id_str))
            
            embedding = model.encode(chunk["content"]).tolist()

            metadata = {
                "source": original_filename,
                "start_line": int(chunk["start_line"]),
                "type": str(chunk.get("type", "code")),
                "function_name": str(chunk.get("function_name", "")),
                "class_name": str(chunk.get("class_name", "")),
                "struct_name": str(chunk.get("struct_name", "")),
                "array_name": str(chunk.get("array_name", "")),
                "test_case_function_name": str(chunk.get("test_case_function_name", ""))
            }
            
            ids_batch.append(chunk_id)
            embeddings_batch.append(embedding)
            documents_batch.append(chunk["content"])
            metadatas_batch.append(metadata)

            if len(ids_batch) >= batch_size:
                try:
                    collection.add(
                        ids=ids_batch,
                        embeddings=embeddings_batch,
                        documents=documents_batch,
                        metadatas=metadatas_batch
                    )
                    print(f"Added batch of {len(ids_batch)} chunks for {original_filename}.")
                except Exception as e:
                    print(f"Error adding batch to ChromaDB for {original_filename}: {e}")
                
                ids_batch = []
                embeddings_batch = []
                documents_batch = []
                metadatas_batch = []
        
        if ids_batch:
            try:
                collection.add(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    documents=documents_batch,
                    metadatas=metadatas_batch
                )
                print(f"Added final batch of {len(ids_batch)} chunks for {original_filename}.")
            except Exception as e:
                print(f"Error adding final batch to ChromaDB for {original_filename}: {e}")

        print(f"Finished processing {original_filename}. Total chunks added: {collection.count()}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

# === Chunk Retriever ===
def retrieve_relevant_chunks(query: str, top_k: int = DEFAULT_TOP_K, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD, filter_type: Optional[str] = None) -> list[dict]:
    """
    Retrieves relevant code chunks from ChromaDB based on a query, with optional filtering by chunk type.
    """
    global collection
    if collection is None:
        print("Error: ChromaDB collection not initialized. Cannot retrieve chunks.")
        return []

    print(f"DEBUG: retrieve_relevant_chunks received query='{query}' with top_k={top_k}, similarity_threshold={similarity_threshold}, filter_type={filter_type}")

    query_embedding = model.encode(query).tolist()

    where_clause = {}
    if filter_type:
        where_clause["type"] = {"$eq": filter_type}

    if where_clause:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'],
            where=where_clause
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

    retrieved_info = []
    if results and results['documents'] and results['distances'] and results['metadatas']:
        for i in range(len(results['documents'][0])):
            doc_content = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]

            if distance < similarity_threshold:
                retrieved_info.append({
                    "content": doc_content,
                    "source": metadata.get("source", "N/A"),
                    "start_line": metadata.get("start_line", -1),
                    "type": metadata.get("type", "code"),
                    "function_name": metadata.get("function_name", ""),
                    "class_name": metadata.get("class_name", ""),
                    "struct_name": metadata.get("struct_name", ""),
                    "array_name": metadata.get("array_name", ""),
                    "test_case_function_name": metadata.get("test_case_function_name", ""),
                    "distance": distance
                })
            else:
                print(f"Skipping chunk due to low similarity (distance {distance:.4f} >= threshold {similarity_threshold:.4f}): {doc_content[:50]}...")

    retrieved_info.sort(key=lambda x: x['distance'])

    unique_retrieved_info = []
    seen_unique_keys = set()
    for item in retrieved_info:
        unique_key = (item['content'], item['source'], item['start_line'], item['type'],
                      item['function_name'], item['class_name'], item['struct_name'],
                      item['array_name'], item['test_case_function_name'])
        if unique_key not in seen_unique_keys:
            unique_retrieved_info.append(item)
            seen_unique_keys.add(unique_key)
            
    print(f"Retrieved {len(unique_retrieved_info)} unique chunks for query: '{query}'")
    for i, chunk in enumerate(unique_retrieved_info):
        name_parts = []
        if chunk.get('function_name'): name_parts.append(f"Func={chunk['function_name']}")
        if chunk.get('class_name'): name_parts.append(f"Class={chunk['class_name']}")
        if chunk.get('struct_name'): name_parts.append(f"Struct={chunk['struct_name']}")
        if chunk.get('array_name'): name_parts.append(f"Array={chunk['array_name']}")
        if chunk.get('test_case_function_name'): name_parts.append(f"TestFunc={chunk['test_case_function_name']}")
        
        name_info = ", ".join(name_parts) if name_parts else 'N/A'

        print(f"  Chunk {i+1}: Source={chunk['source']}, Line={chunk['start_line']}, Type={chunk['type']}, {name_info}, Distance={chunk['distance']:.4f}")
        print(f"  --- Content Start ---\n{chunk['content']}\n  --- Content End ---")

    return unique_retrieved_info

# Example of how you would initialize client and collection in your main application:
# from chromadb.config import Settings
# client = chromadb.PersistentClient(path="./chroma_db") # or chromadb.Client() for in-memory
# collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION_CHROMA)