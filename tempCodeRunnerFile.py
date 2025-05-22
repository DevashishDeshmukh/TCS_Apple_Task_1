import httpx

resp = httpx.post("http://localhost:11434/api/generate", json={
    "model": "tinyllama",
    "prompt": "Say hello",
    "stream": False
})

print(resp.status_code)
print(resp.text)
