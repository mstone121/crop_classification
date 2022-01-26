import os
import sys
import requests
import re

from io import BytesIO
from zipfile import ZipFile

# This script should work as long as google doesn't change things up too much. (2022-01-26)

baseUrl = "https://drive.google.com"

if os.path.isdir("data"):
    print("Data directory already exists. Exiting...")
    sys.exit()

print("Getting cookie...")
initial_response = requests.get(
    baseUrl + "/uc?export=download&id=17kpo8K_t6jS3ZzvWt25WEMbwq_FsM-SF"
)

match = re.search(r'href="(/uc\?export=download.*?)"', initial_response.text)
if match is None:
    print("Could not extract confirm download URL from initial response")
    sys.exit()

print("Downloading file...")
zip_file_response = requests.get(
    baseUrl + match.group(1).replace("&amp;", "&"),
    cookies=initial_response.cookies,
)

with ZipFile(BytesIO(zip_file_response.content)) as zip_file:
    print("Extracting zip file...")
    zip_file.extractall("data")
