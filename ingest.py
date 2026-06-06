import os
import pickle
import faiss
from git import Repo
from sentence_transformers import SentenceTransformer

def clone_target_repo(repo_url, target_dir="./cloned_repo"):
    """Clones a remote open-source repository locally if it doesn't exist."""
    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        print(f"Cloning remote repository: {repo_url}...")
        Repo.clone_from(repo_url, target_dir)
        print("Cloning complete.")
    else:
        print("Target repository already exists locally.")

def chunk_python_code_by_functions(directory):
    """
    Parses Python files structurally. Instead of raw character counting, 
    it splits the code into chunks consisting of whole functions/classes.
    """
    chunks = []
    metadata = []
    
    for root, dirs, files in os.walk(directory):
        # Skip internal git metadata directories
        if '.git' in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                
                current_chunk = []
                current_context = f"# File: {relative_path}\n"
                
                for line in lines:
                    # Detect structural boundaries (Function or Class definitions)
                    if line.startswith("def ") or line.startswith("class "):
                        # If we already accumulated a block of code, save it as a chunk
                        if len(current_chunk) > 3: 
                            chunk_text = current_context + "".join(current_chunk)
                            chunks.append(chunk_text)
                            metadata.append({"source_file": relative_path})
                        
                        # Reset for the new function/class block
                        current_chunk = [line]
                    else:
                        current_chunk.append(line)
                
                # Catch the final trailing block of the file
                if current_chunk:
                    chunk_text = current_context + "".join(current_chunk)
                    chunks.append(chunk_text)
                    metadata.append({"source_file": relative_path})
                    
    return chunks, metadata

def main():
    print("--- Starting Source Code Ingestion Pipeline ---")
    
    # We will use a lightweight, clean public python utility library as our target
    target_url = "https://github.com/pallets/click.git" 
    local_repo_dir = "./cloned_repo"
    
    # 1. Fetch data layer assets
    clone_target_repo(target_url, local_repo_dir)
    
    # 2. Extract code-specific chunks structurally
    print("Parsing codebase and extracting functional blocks...")
    chunks, metadata = chunk_python_code_by_functions(local_repo_dir)
    print(f"Extracted {len(chunks)} distinct functional code blocks.")
    
    if not chunks:
        print("No valid source files parsed. Exiting.")
        return

    # 3. Model Initialization
    print("Loading local embedding engine (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 4. Generate vectors
    print("Generating mathematical vectors across codebase elements...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # 5. Build FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 6. Serialized Persistent Cache Commitment
    print("Committing structural code vectors to disk...")
    faiss.write_index(index, "code_vector_index.bin")
    
    with open("code_metadata.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)
        
    print("\n[SUCCESS] Codebase index completely initialized offline!")

if __name__ == "__main__":
    main()