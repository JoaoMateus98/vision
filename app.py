from flask import Flask, render_template
import secrets
from text_detector import *

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = secrets.token_hex(16)  # Generates a secure random string

@app.route('/', methods=['GET'])
def get_image():
    detect_text()
    return render_template("display_image.html")

if __name__ == '__main__':
    app.run(debug=True)