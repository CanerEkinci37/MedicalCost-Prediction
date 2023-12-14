from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import util

app = FastAPI()


@app.get("/get_region_names")
async def get_region_names():
    util.load_artifacts()
    return util.get_region_names()


@app.post("/predict_medical_cost")
async def predict_medical_cost(request: Request):
    util.load_artifacts()
    form_data = await request.form()
    age = form_data.get("age")
    bmi = float(form_data.get("bmi"))
    children = int(form_data.get("children"))
    ismale = int(form_data.get("ismale"))
    issmoker = int(form_data.get("issmoker"))
    region = form_data.get("region")

    return JSONResponse(
        {
            "predicted_medical_cost": util.predict_medical_cost(
                age, bmi, children, ismale, issmoker, region
            )
        }
    )
