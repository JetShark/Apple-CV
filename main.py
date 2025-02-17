from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
import io

app = FastAPI(title="Apple Image Processor")

def process_image(image: Image.Image) -> Image.Image:
    """
    Stub for image processing. Currently just returns the original image.
    In implementation, this will contain the apple detection logic.
    """
    return image

@app.post("/is_apple", response_class=Response)
async def is_apple(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded image and return the processed version
    """
    # Read and validate the image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        return Response(
            content=str(e),
            media_type="text/plain",
            status_code=400
        )
    
    # Process the image
    processed_image = process_image(image)
    
    # Convert processed image to bytes
    img_byte_array = io.BytesIO()
    processed_image.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()
    
    return Response(
        content=img_byte_array,
        media_type="image/png"
    )