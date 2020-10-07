from typing import List
import uvicorn
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import HTMLResponse

import logging
app = FastAPI()


# @app.post("/json_response/")
# async def create_json_response(files: List[UploadFile] = File(...)):
#     response = None
#     try:
#         data = request_disptacher(files[0], extension="json")
#         response = Response(content=data, media_type="application/json")
#     except Exception as error:
#         logging.error("Error while uploading file ", str(error))
#         response = {"Error ": str(error)}
#
#     return response


@app.get("/")
async def main():
    content = """
            <body>
            <form enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit" formaction="/json_response/" value="Generate JSON Response">
            <input type="submit" formaction="/xml_response/" value="Generate XML Response">
            </form>
            </body>
            """
    return HTMLResponse(content=content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)%