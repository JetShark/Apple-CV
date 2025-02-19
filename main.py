from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI(title="Apple Image Processor")

def process_image(image: Image.Image) -> Image.Image:
    """
    Stub for image processing. Currently just returns the original image.
    In implementation, this will contain the apple detection logic.
    """
    # use HSV to detect red and green apples
    # convert image to HSV
    # image = image.convert('HSV')

    # convert image to numpy array
    original = np.array(image)
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

    # define range of red color in HSV
    lower_red = np.array([0,50,50])
    upper_red = np.array([16,255,255])
    # also detect red color in the upper range
    lower_red2 = np.array([170,50,50])
    upper_red2 = np.array([180,255,255])

    # define range of green color in HSV
    lower_green = np.array([30,100,100])
    upper_green = np.array([70,255,255])

    # Threshold the HSV image to get only red colors
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Threshold the HSV image to get only green colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res_red = cv2.bitwise_and(hsv,hsv, mask= mask_red)
    res_red2 = cv2.bitwise_and(hsv,hsv, mask= mask_red2)
    res_green = cv2.bitwise_and(hsv,hsv, mask= mask_green)

    # merge both red and green masks into one image
    res = cv2.add(res_red, res_red2)
    res = cv2.add(res, res_green)
    
    # apply morphological operations to remove noise
    kernel = np.ones((5,5),np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel) # remove noise
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel) # fill in the holes

    # find contours in the mask and draw rectangles around the apples
    gray_res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
    gray_res = cv2.cvtColor(gray_res, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray_res, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    # apply Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=150, param1=40, param2=50, minRadius=30, maxRadius=150)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(res, (x, y), r, (120, 100, 255), 4)
            # cv2.rectangle(res, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    contours, _ = cv2.findContours(
        gray_res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c_num = 0
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        c_num += 1
        cv2.rectangle(res, (x, y), (x+w, y+h), (50,200,200), 2)
        cv2.putText(res, "#{}".format(c_num), (int(x) - 10, int(y)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,100,100), 2)

    # convert image back to RGB
    image = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)

    # convert numpy array to PIL image
    image = Image.fromarray(image)

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