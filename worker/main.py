import json

from js import ReadableStream, TextEncoder
from pyodide.ffi import create_proxy, to_js
from tone_http import (
    api_dispatch,
    response_headers,
    stream_response_headers,
    wants_stream,
)
from workers import Response, WorkerEntrypoint


def json_response(payload, status=200):
    return Response(
        json.dumps(payload, separators=(",", ":")),
        status=status,
        headers=response_headers(),
    )


def streaming_response(request, env):
    encoder = TextEncoder.new()

    async def start(controller):
        def emit(event):
            line = json.dumps(event, separators=(",", ":")) + "\n"
            controller.enqueue(encoder.encode(line))

        try:
            result = await api_dispatch(request, env, emit)
            if result is None:
                emit({"type": "error", "status": 404, "error": "Not found"})
            else:
                payload, status = result
                if status == 200:
                    emit({"type": "complete", "status": status, "result": payload})
                else:
                    emit(
                        {
                            "type": "error",
                            "status": status,
                            "error": payload.get("error", "Request failed"),
                        }
                    )
        finally:
            controller.close()

    stream = ReadableStream.new(to_js({"start": create_proxy(start)}))
    return Response(stream, status=200, headers=stream_response_headers())


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        if wants_stream(request):
            return streaming_response(request, self.env)
        result = await api_dispatch(request, self.env)
        if result is None:
            return await self.env.ASSETS.fetch(request)
        payload, status = result
        return json_response(payload, status)
