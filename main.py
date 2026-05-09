
from aiohttp import web
import asyncio
import lib.ngram as ngram
import glob

async def handler(request: web.Request):
    seed = hash(request.path)

    response = web.StreamResponse()
    response.content_type = "text/plain"
    await response.prepare(request)
    end_token = ngram_model.tokenize("</START>")[0]
    tokens = ngram_model.tokenize("<START>")
    for _ in range(100):
        tokens.append(ngram_model.get_next(tokens, seed))
        if tokens[-1] == end_token:
            break
        b = bytearray(tokens[-1], 'utf-8')
        await response.write(b)
        await asyncio.sleep(0.1)
    return response

if __name__ == "__main__":

    global ngram_model
    ngram_model = ngram.Model()
    ngram_model.train(list(glob.glob("data/*.*")))

    app = web.Application()
    app.add_routes([web.get(r"/{key:.+}", handler)])
    web.run_app(app)
