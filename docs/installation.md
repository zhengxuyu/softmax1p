# Installation

## From PyPI

softmax1p has no required dependencies. Pick the extra for your framework, or install the framework yourself:

```bash
pip install softmax1p             # no framework pulled in
pip install "softmax1p[torch]"    # + torch
pip install "softmax1p[jax]"      # + jax
pip install "softmax1p[tf]"       # + tensorflow
```

Python 3.9 or newer is required.

The package installs five top-level modules: `softmax_torch`, `softmax_jax`, `softmax_tf`, `vocab_parallel_torch` and `vocab_parallel_jax`. A module only imports its own framework, so `import softmax_torch` works without JAX or TensorFlow installed.

## Copying a single file

Each module is a single file with no dependency beyond its framework. You can also copy it straight into your project:

```bash
curl -O https://raw.githubusercontent.com/zhengxuyu/softmax1p/main/softmax_torch.py
```

## From source

```bash
git clone https://github.com/zhengxuyu/softmax1p
cd softmax1p
pip install -e ".[torch]"
pip install pytest && python -m pytest tests   # test files skip frameworks that are not installed
```
