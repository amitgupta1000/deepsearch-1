import os
## Removed hardcoded GOOGLE_APPLICATION_CREDENTIALS; GCP uses service account authentication

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
            "message": "Saving and retrieving test document!",
            "extra_field": "This is a test value."
        }

        # Save document
        db.collection(test_collection).document(test_doc_id).set(test_data)
        logger.info(f"Successfully saved document: {test_doc_id}")
        print(f"Document {test_doc_id} saved.")

        # Retrieve document
        doc = db.collection(test_collection).document(test_doc_id).get()
        if doc.exists:
            data = doc.to_dict()
            logger.info(f"Retrieved document: {data}")
            print(f"Retrieved document: {data}")
            print("Firestore save/retrieve test PASSED.")
        else:
            logger.error("Document not found after save.")
            print("Firestore save/retrieve test FAILED.")
    except Exception as e:
        logger.error(f"Firestore save/retrieve test failed: {e}")
        print(f"Firestore save/retrieve test FAILED: {e}")

if __name__ == "__main__":
    test_firestore_connectivity()
