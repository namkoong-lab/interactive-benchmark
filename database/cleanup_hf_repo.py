#!/usr/bin/env python3
"""
Clean up HuggingFace repository by deleting all files.
Use this before doing a fresh upload with flat structure.
"""
from huggingface_hub import HfApi

# Configuration
REPO_ID = "gilberty005/personas-product-database"

def cleanup_repository():
    """Delete all files from HuggingFace repository."""
    
    print("="*70)
    print("  Clean Up HuggingFace Repository")
    print("="*70)
    print()
    print(f"⚠️  WARNING: This will delete ALL files from:")
    print(f"   {REPO_ID}")
    print()
    
    confirmation = input("Type 'DELETE' to confirm: ")
    if confirmation != "DELETE":
        print("❌ Aborted - no files were deleted")
        return
    
    print()
    print(f"🧹 Cleaning up repository: {REPO_ID}")
    print()
    
    api = HfApi()
    
    # Get list of all files in the repository
    try:
        repo_files = api.list_repo_files(
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        print(f"📋 Found {len(repo_files)} files/paths to delete:\n")
        
        # Filter out .gitattributes (keep it)
        files_to_delete = [f for f in repo_files if f != ".gitattributes"]
        
        if not files_to_delete:
            print("✅ Repository is already clean (only .gitattributes remains)")
            return
        
        # Delete each file
        deleted_count = 0
        failed_files = []
        
        for file_path in files_to_delete:
            try:
                print(f"   Deleting: {file_path}...")
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Clean up: delete {file_path}"
                )
                deleted_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to delete {file_path}: {e}")
                failed_files.append(file_path)
        
        print()
        print("="*70)
        print(f"✅ Cleanup Complete!")
        print("="*70)
        print(f"\n📊 Summary:")
        print(f"   Deleted: {deleted_count} files")
        if failed_files:
            print(f"   Failed: {len(failed_files)} files")
            print(f"   Failed files: {failed_files}")
        print()
        print(f"🚀 Ready for clean upload!")
        print(f"   Next steps:")
        print(f"   1. python export_to_parquet.py  (if needed)")
        print(f"   2. python upload_parquet_to_hf.py")
        
    except Exception as e:
        print(f"❌ Error accessing repository: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Verify repository exists and you have write access")

if __name__ == "__main__":
    cleanup_repository()

