import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.responses import RedirectResponse

from financeqa.app.routers.chat import chat
from financeqa.app.routers.health import health

load_dotenv()

app = FastAPI()
app.include_router(chat.router)
app.include_router(health.router)


@app.get("/")
async def redirect():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
