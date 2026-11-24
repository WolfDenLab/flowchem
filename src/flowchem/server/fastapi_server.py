"""FastAPI server for devices control."""

import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from importlib.metadata import metadata, version
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from flowchem.components.device_info import DeviceInfo
from flowchem.devices.flowchem_device import RepeatedTaskInfo
from flowchem.server.command_log import CommandLogger
from flowchem.vendor.repeat_every import repeat_every

# Default location for the local run_id/command audit trail (repo root).
_DEFAULT_COMMAND_LOG_PATH = Path(__file__).resolve().parents[3] / "command_log.db"

# Paths excluded from the command log to cut noise from docs/schema browsing.
_COMMAND_LOG_EXCLUDED_PREFIXES = ("/docs", "/openapi.json", "/redoc", "/favicon.ico")


class _RunStartRequest(BaseModel):
    label: str
    manifest: dict[str, Any] | None = None


def _decode_for_log(raw: bytes) -> str | None:
    if not raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"<binary {len(raw)} bytes>"


class FastAPIServer:
    def __init__(
        self,
        filename: str = "",
        host: str = "127.0.0.1",
        port: int = 8000,
        command_log_path: str | Path = _DEFAULT_COMMAND_LOG_PATH,
    ) -> None:
        self._startup_tasks: list = []
        self.command_log = CommandLogger(command_log_path)

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            for task in self._startup_tasks:
                await task()
            yield

        # Create FastAPI app
        self.app = FastAPI(
            title=f"Flowchem - {filename}",
            description=metadata("flowchem")["Summary"],
            version=version("flowchem"),
            license_info={
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT",
            },
            lifespan=lifespan,
        )
        self.base_url = rf"http://{host}:{port}"
        self.configuration_dict: dict[str, Any] = {}

        self._add_root_redirect()
        self._add_configuration_retrieve()
        self._add_run_endpoints()
        self._add_command_log_middleware()

        logger.debug("HTTP ASGI server app created")

    def _add_root_redirect(self) -> None:
        def home_redirect_to_docs():
            """Redirect root to `/docs` to enable interaction w/ API."""
            return RedirectResponse(url="/docs")

        self.app.add_api_route(
            "/",
            home_redirect_to_docs,
            methods=["GET"],
            include_in_schema=False,
        )

    def _add_configuration_retrieve(self) -> None:
        @self.app.get(
            "/startup_config",
            tags=["system"],
        )
        def config():
            """
            Return the startup config server configuration as a dictionary.
            This endpoint provides configuration data used by the server,
            such as device settings and parameters.
            """
            return self.configuration_dict

    def _add_run_endpoints(self) -> None:
        @self.app.post("/runs/start", tags=["system"])
        def start_run(request: _RunStartRequest):
            """Start a new run_id, tagging it with the agent's stated intent."""
            run_id = self.command_log.start_run(
                label=request.label, manifest=request.manifest
            )
            return {"run_id": run_id}

        @self.app.post("/runs/{run_id}/end", tags=["system"])
        def end_run(run_id: str, status: str = "completed"):
            """Close a run_id. `status` is a free-form outcome label."""
            if not self.command_log.end_run(run_id, status=status):
                raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
            return {"run_id": run_id, "status": status}

        @self.app.get("/runs/{run_id}", tags=["system"])
        def get_run(run_id: str):
            """Return a run's metadata plus its full command_log."""
            run = self.command_log.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
            run["commands"] = self.command_log.get_run_commands(run_id)
            return run

    def _add_command_log_middleware(self) -> None:
        @self.app.middleware("http")
        async def log_commands(request: Request, call_next):
            if request.url.path.startswith(_COMMAND_LOG_EXCLUDED_PREFIXES):
                return await call_next(request)

            start = time.monotonic()
            raw_request_body = await request.body()
            request_body = _decode_for_log(raw_request_body)
            run_id = request.headers.get("x-run-id")

            try:
                response = await call_next(request)
            except Exception as exc:
                self.command_log.log_command(
                    run_id=run_id,
                    method=request.method,
                    path=request.url.path,
                    query_params=str(request.query_params) or None,
                    request_body=request_body,
                    status_code=None,
                    response_body=None,
                    duration_ms=(time.monotonic() - start) * 1000,
                    error=repr(exc),
                )
                raise

            response_chunks = [chunk async for chunk in response.body_iterator]
            response_body = b"".join(response_chunks)

            self.command_log.log_command(
                run_id=run_id,
                method=request.method,
                path=request.url.path,
                query_params=str(request.query_params) or None,
                request_body=request_body,
                status_code=response.status_code,
                response_body=_decode_for_log(response_body),
                duration_ms=(time.monotonic() - start) * 1000,
            )

            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

    def add_background_tasks(
        self,
        repeated_tasks: RepeatedTaskInfo | Iterable[RepeatedTaskInfo],
    ):
        """Schedule repeated tasks to run upon server startup.

        Note: each task is bound via a default argument to avoid the classic
        Python loop-closure pitfall where all closures would otherwise share
        the last value of the loop variable.
        """
        if isinstance(repeated_tasks, RepeatedTaskInfo):
            repeated_tasks = (repeated_tasks,)

        for seconds_delay, task in repeated_tasks:
            # `t=task` captures the current task by value, not by reference.
            @repeat_every(seconds=seconds_delay)
            async def _repeated_task(t=task):
                logger.debug("Running repeated task...")
                await t()

            self._startup_tasks.append(_repeated_task)

    def add_device(self, device):
        """Add device to server."""
        # Add components URL to device_info
        components_w_url = {
            component.name: f"{self.base_url}/{device.name}/{component.name}"
            for component in device.components
        }
        device.device_info.components = components_w_url

        # Base device endpoint
        device_root = APIRouter(prefix=f"/{device.name}", tags=[device.name])
        device_root.add_api_route(
            "/",
            device.get_device_info,
            methods=["GET"],
            response_model=DeviceInfo,
        )
        self.app.include_router(device_root)

        # Add repeated tasks for device if any
        if tasks := device.repeated_task():
            self.add_background_tasks(tasks)

        # add device components
        logger.debug(f"Device '{device.name}' has {len(device.components)} components")
        for component in device.components:
            self.app.include_router(component.router, tags=component.router.tags)
            logger.debug(f"Router <{component.router.prefix}> added to app!")
