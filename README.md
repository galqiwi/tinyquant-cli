# TinyQuant CLI

A command-line interface for [TinyQuant](https://github.com/galqiwi/tinyquant) to easily quantize Large Language Models **off GPU**.

## Installation

```bash
uv pip install tinyquant-cli

```

## Usage

Run the quantization process by specifying the model, method, and output path:

```bash
tinyquant-cli --model meta-llama/Llama-3.2-1B --method hqq --save_path ./quantized_model

```

**Common arguments:**

* `--model`: HuggingFace model ID or local path.
* `--method`: Quantization method (e.g., `hqq`).
* `--save_path`: Directory to save the quantized model.

For a full list of options:

```bash
tinyquant-cli --help

```

## Development

To install in development mode:

```bash
uv pip install -e .

```

## License

MIT
