import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import uvicorn
from fastapi import FastAPI, UploadFile
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


def compare_images(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    value, ssim_map = ssim(gray1, gray2, multichannel=False, full=True)
    print("ssim value in this:", value)
    diff = (ssim_map < 0.95).astype("uint8") * 255
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    differences = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        differences.append({"x": x, "y": y, "width": w, "height": h})
    print("differences are:", differences)
    return differences


origins = ["*"]

middleware = [
    Middleware(CORSMiddleware, allow_origins=origins)
]


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app = FastAPI(middleware=middleware)


@app.get("/")
def defaultfn():
    return {"data": "test"}


@app.get("/samplejson")
def jsonfn():
    return {
        "differences": [
            {
                "x": 891,
                "y": 2988,
                "width": 28,
                "height": 28
            },
            {
                "x": 2075,
                "y": 2760,
                "width": 89,
                "height": 49
            },
            {
                "x": 2008,
                "y": 2760,
                "width": 58,
                "height": 39
            },
            {
                "x": 2565,
                "y": 2591,
                "width": 29,
                "height": 40
            },
            {
                "x": 2006,
                "y": 2318,
                "width": 55,
                "height": 40
            },
            {
                "x": 2070,
                "y": 2317,
                "width": 207,
                "height": 50
            },
            {
                "x": 2556,
                "y": 2191,
                "width": 66,
                "height": 167
            },
            {
                "x": 2151,
                "y": 2117,
                "width": 6,
                "height": 1
            },
            {
                "x": 2089,
                "y": 2109,
                "width": 7,
                "height": 8
            },
            {
                "x": 2551,
                "y": 2104,
                "width": 2,
                "height": 12
            },
            {
                "x": 2027,
                "y": 2104,
                "width": 1,
                "height": 13
            },
            {
                "x": 2027,
                "y": 2087,
                "width": 1,
                "height": 2
            },
            {
                "x": 2558,
                "y": 2086,
                "width": 60,
                "height": 38
            },
            {
                "x": 2031,
                "y": 2085,
                "width": 515,
                "height": 39
            },
            {
                "x": 2749,
                "y": 1932,
                "width": 1,
                "height": 1
            },
            {
                "x": 2746,
                "y": 1907,
                "width": 1,
                "height": 1
            },
            {
                "x": 2745,
                "y": 1905,
                "width": 1,
                "height": 1
            },
            {
                "x": 2739,
                "y": 1902,
                "width": 3,
                "height": 1
            },
            {
                "x": 2722,
                "y": 1902,
                "width": 3,
                "height": 30
            },
            {
                "x": 2725,
                "y": 1901,
                "width": 259,
                "height": 32
            },
            {
                "x": 3212,
                "y": 1621,
                "width": 11,
                "height": 12
            },
            {
                "x": 2565,
                "y": 1515,
                "width": 29,
                "height": 40
            },
            {
                "x": 1201,
                "y": 1515,
                "width": 29,
                "height": 40
            }
        ]
    }


@app.post("/compare")
async def compare(
        image1: UploadFile, image2: UploadFile
):
    print("start")
    image1 = np.frombuffer(await image1.read(), dtype="uint8")
    image2 = np.frombuffer(await image2.read(), dtype="uint8")
    print(len(image1))
    image1 = cv2.imdecode(image1, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imdecode(image2, cv2.IMREAD_UNCHANGED)
    print("images shape:", image1.shape, image2.shape)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    value, ssim_map = ssim(gray1, gray2, multichannel=False, full=True)
    print("ssim value in this:", value)
    diff = (ssim_map < 0.95).astype("uint8") * 255
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    differences = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        differences.append({"x": x, "y": y, "width": w, "height": h})
    print("differences are:", differences)
    return differences
    # differences = compare_images(image1, image2)
    # return {"differences": differences}
