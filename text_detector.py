import os
import io
from google.cloud import vision, storage
from PIL import Image, ImageDraw, UnidentifiedImageError


def detect_text():
    # Initialize the Google Vision API and Storage clients
    vision_client = vision.ImageAnnotatorClient()
    storage_client = storage.Client()

    # Get the bucket name from the environment variable set in app.yaml
    bucket_name = os.environ.get('BUCKET_NAME')

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # List all blobs in the bucket (images)
    blobs = list(bucket.list_blobs())

    # Identify blobs with "__boxed.png" in the name
    boxed_blobs = {blob.name.split('__boxed.png')[0] for blob in blobs if "__boxed.png" in blob.name}

    # Filter out blobs with "__boxed.png" in their names and also skip those that have a corresponding "__boxed.png"
    image_blobs = [
        blob for blob in blobs
        if "__boxed.png" not in blob.name and
           blob.name.split('.')[0] not in boxed_blobs and  # Skip images with a corresponding "__boxed.png"
           blob.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))
    ]

    new_blob_uris = []

    # Add all blobs with "__boxed.png" to new_blob_uris (skip processing)
    for blob in blobs:
        if "__boxed.png" in blob.name:
            print(f'Skipping already processed image: {blob.name}')
            new_blob_uris.append(blob.public_url)

    for blob in image_blobs:
        print(f'\nProcessing file: {blob.name}')

        # Read the image content from GCS
        try:
            content = blob.download_as_bytes()
        except Exception as e:
            print(f"Error downloading image {blob.name}: {e}")
            continue

        # Try to open the image with PIL to check if it's a valid image
        try:
            image = Image.open(io.BytesIO(content))
        except UnidentifiedImageError:
            print(f"Error: {blob.name} is not a valid image file.")
            continue

        # Prepare the image for the Google Vision API
        image = vision.Image(content=content)

        # Perform text detection
        response = vision_client.document_text_detection(image=image)
        texts = response.text_annotations
        if texts:
            print(f'\n"{texts[0].description}"')

            # Load image into PIL for drawing
            image_stream = io.BytesIO(content)
            with Image.open(image_stream) as img:
                draw = ImageDraw.Draw(img)

                # Draw bounding boxes around detected text
                for text in texts[1:]:  # Skip the first element which is the entire text block
                    vertices = text.bounding_poly.vertices
                    box = [(vertex.x, vertex.y) for vertex in vertices]
                    draw.line(box + [box[0]], width=2, fill="red")

                # Prepare image for upload
                output_image_stream = io.BytesIO()
                img.save(output_image_stream, format='PNG')
                output_image_stream.seek(0)

                # Create a new blob for the output image with a modified name
                output_blob_name = f'{os.path.splitext(blob.name)[0]}__boxed.png'
                output_blob = bucket.blob(output_blob_name)

                # Upload the modified image to the bucket
                output_blob.upload_from_file(output_image_stream, content_type='image/png')
                print(f"Saved image with bounding boxes to {output_blob_name} in bucket {bucket_name}")

                # Make the blob publicly accessible
                output_blob.make_public()

                # Add the URI of the new blob to the list
                new_blob_uris.append(output_blob.public_url)
        else:
            print('No text found in the image.')

        # Check for any errors in the response
        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(response.error.message)
            )

    return new_blob_uris

