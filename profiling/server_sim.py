import torch

VOCAB = {f"tok{i}" for i in range(5000)}

def tokenize(text):
    return [w for w in text.split() if w in VOCAB]

def build_batch(requests):
    return [tokenize(r) for r in requests]

def forward(batch):
    x = torch.randn(len(batch), 512, 512)
    w = torch.randn(512, 512)
    return (x @ w).softmax(dim=-1)

def handle_requests(requests):
    batch = build_batch(requests)
    return forward(batch)

requests = ["tok4999 tok4998 tok4997 tok12 tok3000"] * 32

for _ in range(200):
    handle_requests(requests)
