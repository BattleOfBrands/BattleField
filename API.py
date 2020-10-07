from typing import List
import uvicorn
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import HTMLResponse
from LogoScout.image_processor import process_image
import json

import logging
app = FastAPI()


@app.post("/process_image/")
async def create_json_response(files: List[UploadFile] = File(...)):
    response = None
    try:
        data = json.dumps(process_image(files[0]))
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)