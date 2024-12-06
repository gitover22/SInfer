import asyncio  # 用于管理异步任务
import importlib
import inspect
import re
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Set

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

import sinfer
import sinfer.envs as envs
from sinfer.sys.arg_utils import AsyncEngineArgs
from sinfer.sys.async_runtime import AsyncLLMRuntime
from sinfer.entrypoints.openai.cli_args import make_arg_parser
from sinfer.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest, ErrorResponse)
from sinfer.entrypoints.openai.serving_chat import OpenAIServingChat
from sinfer.entrypoints.openai.serving_completion import OpenAIServingCompletion
from sinfer.logger import init_logger
from sinfer.usage.usage_lib import UsageContext

TIMEOUT_KEEP_ALIVE = 5   # 连接的保活超时时间 second

openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
logger = init_logger(__name__)

_running_tasks: Set[asyncio.Task[Any]] = set()  # 用于跟踪异步任务


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log(): # 每隔 10 秒记录一次推理引擎的统计信息
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield

# 实例化FastAPI
app = fastapi.FastAPI(lifespan=lifespan)

# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)








@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

# 推理引擎健康检查
@app.get("/health")
async def health() -> Response:
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)

# 展示可用模型
@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())

# 返回版本信息
@app.get("/version")
async def show_version():
    ver = {"version": sinfer.__version__}
    return JSONResponse(content=ver)

# 聊天补全 ( 最常用 )
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())

# 文本补全
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    parser = make_arg_parser()
    args = parser.parse_args()
    app.add_middleware(  # 添加 CORS 中间件
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,  # 是否允许发送凭证（如 Cookies
        allow_methods=args.allowed_methods,   # 允许的 HTTP 方法
        allow_headers=args.allowed_headers,   # 允许的请求头
    )

    if token := envs.SINFER_API_KEY or args.api_key:

        @app.middleware("http") # 定义一个 HTTP 请求的中间件
        async def authentication(request: Request, call_next):  # 异步认证中间件
            root_path = "" if args.root_path is None else args.root_path
            if not request.url.path.startswith(f"{root_path}/v1"): # 如果请求路径不以 /v1 开头，直接处理请求
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request) # 如果认证通过，继续处理请求

    for middleware in args.middleware: # 遍历命令行参数中的 middleware 列表，加载自定义的中间件
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported) 
        elif inspect.iscoroutinefunction(imported): # 如果是协程函数，作为 HTTP 中间件添加
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    logger.info("SInfer API server version %s", sinfer.__version__)
    logger.info("args: %s", args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]
    engine_args = AsyncEngineArgs.from_cli_args(args)  # 创建异步AsyncEngineArgs实例
    engine = AsyncLLMRuntime.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER) # 创建AsyncLLMEngine实例
    openai_serving_chat = OpenAIServingChat(engine, served_model_names,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model_names, args.lora_modules)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
