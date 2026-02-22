import sys

def test_python_version():
    assert sys.version_info >= (3, 11), f"Need Python 3.11 or higher, currently using {sys.version_info}"

def test_torch_mps():
    import torch
    assert torch.backends.mps.is_available(), "MPS not available"
    x = torch.randn(3, 3, device="mps")
    y = torch.randn(3, 3, device="mps")
    z = x @ y #matrix multiply using MPS
    assert z.device.type == "mps"

def test_tiktoken():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode("Hello, world!")
    assert len(tokens) > 0

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
