from io import BytesIO
import base64
import fitz
from config import settings
from utils.shared import databricks_api
import base64

def download_file_from_dbfs(dbfs_path: str):
    offset = 0
    binary_content = bytearray()
    while True:
        response = databricks_api(
            "/api/2.0/dbfs/read",
            "get",
            data={"path": dbfs_path, "offset": offset, "length": 1048576}  # 1 MB chunks
        )
        content = response.json().get("data")
        if not content:
            break  # No more data
        
        # Decode the base64 content and add it to binary_content
        try:
            decoded_content = base64.b64decode(content)
            binary_content.extend(decoded_content)
        except Exception as e:
            raise Exception(f"Failed to decode base64 content: {e}")

        offset += 1048576  # Move to the next chunk
    # Check if the file is a PDF, then read its content
    if dbfs_path.endswith(".pdf"):
        return read_pdf_content(BytesIO(binary_content))
    else:
        return binary_content.decode()  # Decode if not a PDF

def read_pdf_content(pdf_data):
    # Open the PDF from the binary data
    try:
        pdf = fitz.open(stream=pdf_data, filetype="pdf")
        text_content = ""
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text_content += page.get_text()
        pdf.close()
        return text_content
    except Exception as e:
        raise Exception(f"Failed to read PDF content: {e}")