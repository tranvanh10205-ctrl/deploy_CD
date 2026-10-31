
from pathlib import Path
import os
import tempfile
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .inference import model_status, predict_audio
from .language_check import check_english


BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

app = FastAPI(title="Speakify Local")

app.mount(
    "/static",
    StaticFiles(directory=BASE / "static"),
    name="static"
)

templates = Jinja2Templates(directory=BASE / "templates")

ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"
}

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/api/health")
async def health():
    return model_status()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()

    # 1. Kiểm tra định dạng file
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Định dạng âm thanh không được hỗ trợ."
        )

    path = None

    try:
        # 2. Lưu file upload thành file tạm
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:
            path = temp_file.name
            total_size = 0

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File âm thanh vượt quá giới hạn 25 MB."
                    )

                temp_file.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="File âm thanh đang trống."
            )

        # 3. Kiểm tra ngôn ngữ bằng Whisper
        check = check_english(path)

        if not check["accepted"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": check["message"],
                    "detected_language": check["language"],
                    "language_probability": check["language_probability"],
                    "transcript": check["transcript"],
                },
            )

        # 4. Chạy mô hình TS-TCC
        result = predict_audio(path)

        # 5. Bổ sung thông tin nhận dạng ngôn ngữ
        result["transcript"] = check["transcript"]
        result["detected_language"] = check["language"]
        result["language_probability"] = check["language_probability"]

        return result

    except HTTPException:
        raise

    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    finally:
        # 6. Xóa file tạm sau khi xử lý
        if path and os.path.exists(path):
            os.remove(path)

        await file.close()
