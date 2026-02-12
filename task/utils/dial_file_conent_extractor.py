import io
from pathlib import Path

import pdfplumber
import pandas as pd
from aidial_client import Dial
from bs4 import BeautifulSoup


class DialFileContentExtractor:

    def __init__(self, endpoint: str, api_key: str):
        self._dial_client = Dial(base_url=endpoint, api_key=api_key)

    def extract_text(self, file_url: str) -> str:
        file = self._dial_client.files.download(file_url)
        file_name = file.filename
        file_content = file.get_content()
        file_extension = Path(file_name).suffix.lower()
        return self.__extract_text(file_content, file_extension, file_name)

    def __extract_text(self, file_content: bytes, file_extension: str, filename: str) -> str:
        """Extract text content based on file type."""
        try:
            if file_extension == ".txt":
                return file_content.decode('utf-8', errors='ignore')
            elif file_extension == ".pdf":
                pdf_content = io.BytesIO(file_content)
                pdf = pdfplumber.open(pdf_content)
                pdf_pages = [page.extract_text() for page in pdf.pages]
                return "\n".join(pdf_pages)
            elif file_extension == ".csv":
                csv_content = file_content.decode('utf-8', errors='ignore')
                csv_buffer = io.StringIO(csv_content)
                return pd.read_csv(csv_buffer, sep=',', engine='python').to_markdown(index=False)
            elif file_extension in ['.html', '.htm']:
                html_content = file_content.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html_content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator='\n', strip=True)
            else:
                return file_content.decode('utf-8', errors='ignore')
        except Exception as ex:
            print(ex)
            return ""
