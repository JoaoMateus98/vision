import io
import os
import logging
from google.cloud import vision, storage
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np

# Get the bucket name from the environment variable set in app.yaml
BUCKET_NAME = os.environ.get('BUCKET_NAME')

# Configure logging
logging.basicConfig(level=logging.INFO)

def initialize_clients():
    """Initialize Google Vision API and Cloud Storage clients."""
    vision_client = vision.ImageAnnotatorClient()
    storage_client = storage.Client()
    return vision_client, storage_client


def get_image_blobs(storage_client):
    """Get a list of image blobs that need to be processed from the bucket."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs())

    # Identify blobs with "__boxed.png" in the name
    boxed_blobs = {blob.name.split('__boxed.png')[0] for blob in blobs if "__boxed.png" in blob.name}

    # Filter out blobs with "__boxed.png" in their names
    image_blobs = [
        blob for blob in blobs
        if "__boxed.png" not in blob.name and
           blob.name.split('.')[0] not in boxed_blobs and
           blob.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))
    ]

    return image_blobs, blobs


def process_blob(blob, vision_client, bucket):
    """Process a single blob, detect text, draw bounding boxes, and upload the modified image."""
    logging.info(f'Processing file: {blob.name}')

    # Read the image content from GCS
    content = blob.download_as_bytes()

    # Prepare the image for the Google Vision API
    image = vision.Image(content=content)

    # First attempt to detect text
    response = vision_client.document_text_detection(image=image)

    # Check for any errors in the response
    if response.error.message:
        logging.error('Vision API error: %s', response.error.message)
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(response.error.message)
        )

    texts = response.text_annotations

    if not texts:
        logging.info('No text detected on first attempt, preprocessing image and trying again.')

        # Preprocess the image and try text detection again
        preprocessed_image = preprocess_image_for_ocr(content)

        # Convert preprocessed image to bytes for Vision API
        with io.BytesIO() as output:
            preprocessed_image.save(output, format="PNG")
            preprocessed_content = output.getvalue()

        preprocessed_image_vision = vision.Image(content=preprocessed_content)
        response = vision_client.document_text_detection(image=preprocessed_image_vision)
        texts = response.text_annotations

    if texts:
        logging.info('Detected text: "%s"', texts[0].description)

        # Draw bounding boxes around detected text
        img_with_boxes = draw_bounding_boxes(content, texts)

        # Upload the image with bounding boxes
        output_blob = upload_processed_image(img_with_boxes, blob, bucket)

        return output_blob.public_url
    else:
        logging.warning('No text found in the image after preprocessing.')
        return None


# image enhancer
def preprocess_image_for_ocr(image_content):
    # Open the original image
    image_stream = io.BytesIO(image_content)
    with Image.open(image_stream) as img:
        # Convert to grayscale
        gray_img = img.convert('L')

        # Apply Gaussian blur to reduce noise
        blurred_img = gray_img.filter(ImageFilter.GaussianBlur(radius=1))

        # Increase contrast
        enhancer = ImageEnhance.Contrast(blurred_img)
        contrasted_img = enhancer.enhance(2.0)

        # Adaptive thresholding using numpy
        img_array = np.array(contrasted_img)
        mean = np.mean(img_array)
        binary_img = np.where(img_array > mean, 255, 0).astype(np.uint8)

        # Convert back to PIL image
        processed_img = Image.fromarray(binary_img)

        # Optionally sharpen the image
        sharpener = ImageEnhance.Sharpness(processed_img)
        sharpened_img = sharpener.enhance(2.0)

        return sharpened_img


def draw_bounding_boxes(image_content, texts):
    """Draw bounding boxes on the image for the detected text."""
    image_stream = io.BytesIO(image_content)
    with Image.open(image_stream) as img:
        draw = ImageDraw.Draw(img)

        # Draw bounding boxes around detected text
        for text in texts[1:]:  # Skip the first element which is the entire text block
            vertices = text.bounding_poly.vertices
            box = [(vertex.x, vertex.y) for vertex in vertices]
            draw.line(box + [box[0]], width=2, fill="red")

    # Return the modified image with bounding boxes
    output_image_stream = io.BytesIO()
    img.save(output_image_stream, format='PNG')
    output_image_stream.seek(0)

    return output_image_stream


def upload_processed_image(image_stream, blob, bucket):
    """Upload the modified image with bounding boxes to the bucket."""
    output_blob_name = f'{os.path.splitext(blob.name)[0]}__boxed.png'
    output_blob = bucket.blob(output_blob_name)

    # Upload the modified image
    output_blob.upload_from_file(image_stream, content_type='image/png')
    logging.info("Saved image with bounding boxes to %s in bucket %s", output_blob_name, BUCKET_NAME)

    # Make the blob publicly accessible
    output_blob.make_public()

    return output_blob


def get_image_uris():
    """Main function to detect text and add bounding boxes to images."""
    vision_client, storage_client = initialize_clients()

    # Get the image blobs
    image_blobs, blobs = get_image_blobs(storage_client)

    image_uris = []

    # Add all blobs with "__boxed.png" to image_uris (skip processing)
    for blob in blobs:
        if "__boxed.png" in blob.name:
            logging.info('Skipping already processed image: %s', blob.name)
            image_uris.append(blob.public_url)

    # Process each image blob
    for blob in image_blobs:
        processed_blob_uri = process_blob(blob, vision_client, storage_client.bucket(BUCKET_NAME))
        if processed_blob_uri:
            image_uris.append(processed_blob_uri)

    return image_uris