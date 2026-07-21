import logging

logger = logging.getLogger("api")


def serve(**kwargs):
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required to run the API server. Install it with: uv sync --extra api")

    from vectoria_lib.api.app import create_app

    port = int(kwargs.get("port", 8000))
    default_upload_url = f"http://localhost:{port}/upload"
    app = create_app(
        index_path=kwargs.get("index_path"),  # optional: pre-built index
        index_dir=kwargs.get("index_dir", "/data/index"),
        upload_dir=kwargs.get("upload_dir", "/data/uploads"),
        upload_url=kwargs.get("upload_url", default_upload_url),
    )

    host = kwargs.get("host", "0.0.0.0")

    logger.info("Starting Vectoria API on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
