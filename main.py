import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import uvicorn
from fastapi import FastAPI, UploadFile

app = FastAPI()


def compare_images(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    value, ssim_map = ssim(gray1, gray2, multichannel=False, full=True)
    diff = (ssim_map < 0.95).astype("uint8") * 255
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    differences = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        differences.append({"x": x, "y": y, "width": w, "height": h})
    return differences


@app.post("/compare")
async def compare(
    image1: UploadFile, image2: UploadFile
):
    image1 = np.frombuffer(await image1.read(), dtype="uint8")
    image2 = np.frombuffer(await image2.read(), dtype="uint8")
    image1 = cv2.imdecode(image1, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imdecode(image2, cv2.IMREAD_UNCHANGED)
    differences = compare_images(image1, image2)
    return {"differences": differences}
