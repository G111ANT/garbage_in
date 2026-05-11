
from aiohttp import web
import asyncio
import lib.ngram as ngram
import glob
import random

async def handler(request: web.Request):
    start = f"""
    <!DOCTYPE html>
    <html lang="en>
    <head>
        <meta charset="UTF-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>{random.choice(ngram_model.vocab)}</title>
    </head>

    <body>
        <p>
    """
    end = f"""
        </p>
        <a href="{random.choice(ngram_model.vocab)}/">{random.choice(ngram_model.vocab)}</a>
        <a href="{random.choice(ngram_model.vocab)}/">{random.choice(ngram_model.vocab)}</a> 
        <a href="{random.choice(ngram_model.vocab)}/">{random.choice(ngram_model.vocab)}</a> 
        <a href="{random.choice(ngram_model.vocab)}/">{random.choice(ngram_model.vocab)}</a> 
    </body>
    </html> 
    """

    response = web.StreamResponse()
    response.content_type = "text/html"
    await response.prepare(request)

    await response.write(bytearray(start, 'utf-8'))

    end_token = ngram_model.tokenize("</START>")[0]
    tokens = ngram_model.tokenize("<START>")
    rng = random.Random()
    rng.seed(hash(request.path))
    for i in range(100):
        tokens.append(ngram_model.get_next(tokens, rng))
        if tokens[-1] == end_token:
            break
        await response.write(bytearray(tokens[-1], 'utf-8'))
        await asyncio.sleep(random.random() * 0.1)

    await response.write(bytearray(end, 'utf-8'))

    await response.write_eof()
    print(f"{request.path}\n{"".join(tokens).replace("\n", " ")}")
    return response

if __name__ == "__main__":

    global ngram_model
    ngram_model = ngram.Model()
    ngram_model.train(list(glob.glob("data/*.*")))

    app = web.Application()
    app.add_routes([web.get(r"/{key:.+}", handler)])
    web.run_app(app)
