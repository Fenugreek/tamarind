"""Utility functions for downloading from a URL to disk."""

import os, requests
from urllib.parse import urlparse

def get_filename(response, file_url, default_filename='downloaded_file'):
    parsed_url = urlparse(file_url)
    filename = os.path.basename(parsed_url.path)
    if not filename or filename == '/':
        filename = default_filename

    return filename


def download_file(file_url, download_path, default_filename='downloaded_file',
                  chunk_size=8192, timeout=10, logger=None):
    """
    Download a file from given url to location.

    Args:
        file_url (str): The URL of file to download.
        
        download_path (str): Local path where the file should be saved.

        default_filename: Name to give file if filename cannot be determined from URL.
        
        chunk_size (int): Size of chunks for streaming download (default 8192 bytes)

        timeout: no. of seconds before request timesout
    
    Returns:
        str: Path to the downloaded file
    
    Raises:
        requests.RequestException: If download fails
        ValueError: If URL is invalid
    """
    
    if not file_url: raise ValueError("file_url cannot be empty")

    try:
        # Make initial request to get file info and handle redirects
        response = requests.get(file_url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        download_path = os.path.join(download_path,
                                     get_filename(response, file_url, default_filename=default_filename))
        os.makedirs(os.path.dirname(os.path.abspath(download_path)), exist_ok=True)
        
        file_size = int(response.headers.get('Content-Length', 0))        
        if logger:
            logger.info(f"Downloading: {download_path} ({file_size / 1024 / 1024:.2f} MB)")

        # Download the file in chunks
        downloaded = 0
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk: continue  # Filter out keep-alive chunks
                f.write(chunk)
                downloaded += len(chunk)                    
                # Show progress if file size is known
                if logger and file_size > 0:
                    progress = (downloaded / file_size) * 100
                    logger.debug(f"\rProgress: {progress:.1f}% ({downloaded:,} / {file_size:,} bytes)", status_line==True)
        return download_path
        
    except requests.exceptions.RequestException as e:
        raise requests.RequestException(f"Failed to download file: {repr(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error during download: {repr(e)}")
