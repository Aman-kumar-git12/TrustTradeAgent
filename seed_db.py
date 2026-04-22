import sys
import os
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from apps.chat_service.services.knowledge_seed import seed_knowledge_data

def main():
    print("--------------------------------------------------")
    print("🚀 TrustTrade AI: Vector Store Seeding")
    print("--------------------------------------------------")
    print("Connecting to MongoDB and loading models...")
    
    start_time = time.time()
    
    try:
        results = seed_knowledge_data(verbose=True)
        
        if "error" in results:
            print(f"❌ ERROR: {results['error']}")
            sys.exit(1)
            
        duration = time.time() - start_time
        
        print("--------------------------------------------------")
        print("✅ Seeding Complete!")
        print(f"📄 Files Processed: {results['processed_files']}")
        print(f"📥 Chunks Inserted/Updated: {results['inserted_chunks']}")
        print(f"⏭️  Chunks Skipped (No Change): {results['skipped_chunks']}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print("--------------------------------------------------")
        print("The vector Store is now ready for production use.")
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
