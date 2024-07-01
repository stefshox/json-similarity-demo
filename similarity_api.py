import json
import os
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from json_similarity_search import JsonSimilarity

json_similarity = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global json_similarity
    json_file = os.environ.get("JSON_FILE")
    apps_data = json.load(open(f"{json_file}.json"))
    json_similarity = JsonSimilarity(apps_data=apps_data)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/similarity")
async def similarity(new_app_file: UploadFile = File(...)):
    new_app_data = json.load(new_app_file.file)
    most_similar_app = json_similarity.similarity_search(new_app_data=new_app_data)
    return json.dumps(most_similar_app)
