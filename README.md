# Baymax Chat API: chatbot for health care assistance
=====================================================================

> "I cannot deactivate until you say that you are satisfied with your care." - Baymax

The Baymax Chat API is a powerful tool designed to provide healthcare assistance and support through conversational AI. It offers the following endpoints:

- **/predict**: Generate text responses to user prompts, simulating conversations with Baymax.
- **/generate**: Obtain personalized healthcare recommendations and information from Baymax.

The Baymax Chat API aims to provide compassionate and reliable healthcare support, inspired by the caring nature of Baymax from Big Hero 6.


# Usage
You can interact with the `/predict` endpoint using Python as shown below:

```python
import requests

url = "<localhost URL where the server is spun>"
headers = {"Content-type": "application/json"}
data = {
   "url": "https://publicly-accessible-image-path.jpg"
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
```
Make sure to replace "https://publicly-accessible-image-path.jpg" with the actual publicly accessible image path(maybe by uploading to google drive) when using the /predict endpoint for image-related queries. This ensures that Baymax can access and analyze the relevant image accurately.


### Requirements
Python - v3.9 minimum

## Setup for usage
* For using a standalone package
1. Set up a virtual environment (or conda environment). Although not mandatory, it is highly
   recommended separating each project's python environment. To create a virtual environment
   in the project directory itself with the native Python environment manager `venv`:
    ```bash
    $ cd /path/to/project/directory
    $ python3 -m venv .venv #sets up a new virtual env called .venv
    ```
   Next, to activate the virtual environment (say `.venv`):
    ```bash
    $ source .venv/Scripts/activate
    ```
2. Install requirements,
    ```
     pip install -r requirements.txt
    ```
3. Spin the FastAPI server,
    ```
     uvicorn main:app
    ```

## Testing

Tests are added in the `tests` dir.
1. To run all tests simply run:
   ```bash 
   $ pytest 
   ```
2. To run all the tests from one directory, use the directory as a parameter to pytest:
   ```bash
   $ pytest tests/my-directory
   ```
3. To run all tests in a file , list the file with the relative path as a parameter to pytest:
   ```bash
   $ pytest tests/my-directory/test_demo.py
   ```
4. To run a set of tests based on function names, the -k option can be used
   For example, to run all functions that have _raises in their name:
   ````shell
   $ pytest -v -k _raises
   ````
