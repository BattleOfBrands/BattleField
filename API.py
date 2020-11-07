import uvicorn
import json
import logging

from typing import List
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import HTMLResponse

from gaming_engine.gaming_engine import Game
from image_processor import ImageProcessor
from PIL import Image


app = FastAPI()


@app.post("/logo_detection/")
async def logo_detection(files: List[UploadFile] = File(...)):
    response = None
    try:
        image_processor = ImageProcessor()
        data = json.dumps({image_processor.detect_logos(files[0])})
        response = Response(content=data, media_type="application/json")
    except Exception as error:
        logging.error("Error while uploading file ", str(error))
        response = {"Error ": str(error)}

    return response


@app.post("/process_image/")
async def process_image(files: List[UploadFile] = File(...)):
    response = None
    try:
        game = Game()
        image = Image.open(files[0].file)
        data = json.dumps(game.process_image(image))
        response = Response(content=data, media_type="application/json")
    except Exception as error:
        logging.error("Error while uploading file ", str(error))
        response = {"Error ": str(error)}

    return response


@app.post("/impact_generation/")
async def impact_generation(files: List[UploadFile] = File(...)):
    response = None
    try:
        data = json.dumps({"TODO": "TODO"})
        response = Response(content=data, media_type="application/json")
    except Exception as error:
        logging.error("Error while uploading file ", str(error))
        response = {"Error ": str(error)}

    return response


@app.get("/")
async def main():
    content = """
            <body>
            <form enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit" formaction="/process_image/" value="Submit">
            </form>
            </body>
            """
    return HTMLResponse(content=content)

@app.get("/logo_detection")
async def main():
    content = """
            <body>
            Logo Detection
            <form enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit" formaction="/logo_detection/" value="Submit">
            </form>
            </body>
            """
    return HTMLResponse(content=content)

@app.get("/impact_generation")
async def main():
    content = """
            <body>
            Brand Impact Computation
            <form enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit" formaction="/impact_generation/" value="Submit">
            </form>
            </body>
            """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
