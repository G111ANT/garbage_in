
from aiohttp import web
import asyncio
import lib.ngram as ngram

async def handler(request: web.Request):
    seed = hash(request.path)

    response = web.StreamResponse()
    response.content_type = "text/plain"
    await response.prepare(request)
    for _ in range(100):
        b = bytearray('line %d\n' % _, 'utf-8')
        await response.write(b)
        await asyncio.sleep(0.1)
    return response

if __name__ == "__main__":

    # Train n-gram

    app = web.Application()
    app.add_routes([web.get(r"/{key:.+}", handler)])
    web.run_app(app)
