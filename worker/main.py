import json

from tone_http import api_dispatch, response_headers
from workers import Response, WorkerEntrypoint


def json_response(payload, status=200):
    return Response(
        json.dumps(payload, separators=(",", ":")),
        status=status,
        headers=response_headers(),
    )


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        result = await api_dispatch(request, self.env)
        if result is None:
            return await self.env.ASSETS.fetch(request)
        payload, status = result
        return json_response(payload, status)
