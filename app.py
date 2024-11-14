from flask import Flask, render_template
from text_detector import detect_text
import secrets

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = secrets.token_hex(16)  # Generates a secure random string

@app.route('/', methods=['GET'])
def get_image():
    new_blob_uris = detect_text()
    return render_template('display_images.html', image_uris=new_blob_uris)

if __name__ == '__main__':
    app.run(debug=True)