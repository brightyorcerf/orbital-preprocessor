import os
import sys
from flask import Flask

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

app = Flask(__name__)

@app.route('/')
def home():
    return "OSP Command Centre Gateway Active. Please run Streamlit locally for full 3D Globe interactions, or check the LLM logs here."