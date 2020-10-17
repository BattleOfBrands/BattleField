import uvicorn
import json
import logging

from typing import List
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import HTMLResponse

from gaming_engine.gaming_engine import Game

app = FastAPI()


@app.post("/process_image/")
async def create_json_response(files: List[UploadFile] = File(...)):
    response = None
    try:
        game = Game()
        data = json.dumps(game.process_image(files[0]))
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
