import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\gen-lang-client-0665888431-038f11096cad.json"

import os
import logging
from google.cloud import firestore
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_firestore_connectivity():
    try:
        db = firestore.Client()
        test_collection = "connectivity_test"
        test_doc_id = f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        test_data = {
            "timestamp": datetime.now().isoformat(),
            "message": "Firestore connectivity test successful!"
        }
        # Write test document
        db.collection(test_collection).document(test_doc_id).set(test_data)
        logger.info(f"Successfully wrote test document: {test_doc_id}")
        # Read back the document
        doc = db.collection(test_collection).document(test_doc_id).get()
        if doc.exists:
            logger.info(f"Read back document: {doc.to_dict()}")
            print("Firestore connectivity test PASSED.")
        else:
            logger.error("Test document not found after write.")
            print("Firestore connectivity test FAILED.")
    except Exception as e:
        logger.error(f"Firestore connectivity test failed: {e}")
        print(f"Firestore connectivity test FAILED: {e}")

if __name__ == "__main__":
    test_firestore_connectivity()
