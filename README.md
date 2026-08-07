ProtEnc: generate protein embeddings the easy way

[ProtEnc](https://github.com/kklemon/ProtEnc) aims to simplify extraction of protein embeddings from various pre-trained models by providing simple APIs and bulk generation scripts for the ever-growing landscape of protein language models (pLMs). Currently, supported models are:

* [ProtTrans](https://github.com/agemagician/ProtTrans) family
* [ESM](https://github.com/facebookresearch/esm)
* [CARP](https://github.com/microsoft/protein-sequence-models)
* AlphaFold (coming soon™)
* [OmegaPLM](https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1) (coming soon™)

Usage
-----

### Installation

```bash
pip install protenc
```

### Python API

```python
import protenc

# List available models
print(protenc.list_models())

# Load encoder model
encoder = protenc.get_encoder('esm2_t30_150M_UR50D', device='cuda')

proteins = [
  'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
  'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE'
]

# encoder(list) yields (index, embedding) tuples for streamed write-by-index
for idx, embed in encoder(proteins, return_format='numpy'):
  print(idx, embed.shape)

# encoder(dict) yields (key, embedding) tuples
proteins_by_id = {'seq1': proteins[0], 'seq2': proteins[1]}
for key, embed in encoder(proteins_by_id, return_format='numpy'):
  print(key, embed.shape)
```

By default, `average_sequence=True` pools per-residue embeddings to a single vector per sequence (mean over the sequence dimension). Pass `average_sequence=False` to keep shape `[L, D]` where `L` is sequence length and `D` is embedding dimensionality.

Multi-layer extraction is available via `encoder.encode_multi_layer(proteins, repr_layers=[6, 12, 24])`, which yields `(index, layer, embedding)` tuples. Pass `repr_layers` to `get_encoder` to configure the default layer set on the loaded model.

### Command-line interface

After installation, use the `protenc` shell command for bulk generation and export of protein embeddings.

```bash
protenc --help
```

Run example:
- batch size 128
- 4 dataloader workers
- multi-GPU via `torch.nn.DataParallel` (`--data_parallel`)
- substitute amino acid wildcards by possible substitutes

```bash
protenc sequences.fasta embeddings.lmdb --model_name esm2_t33_650M_UR50D --data_parallel --batch_size 128 --num_workers 4 --substitute_wildcards
```

Unlike the Python API, the CLI does not pool embeddings by default. Pass `--compute_mean` (alias `--pool`) to average per-residue outputs along the sequence axis.

Extract representations from multiple transformer layers in one forward pass with `--repr_layers` (writes one output file per layer with a `_layer<N>` suffix):

```bash
protenc sequences.fasta embeddings.lmdb --model_name esm2_t33_650M_UR50D --repr_layers 12 24 33
```

By default, input and output formats are inferred from the file extensions.

**Example**

Generate protein embeddings using the ESM2 650M model for sequences provided in a [FASTA](https://en.wikipedia.org/wiki/FASTA_format) file and write embeddings to an [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database):

```bash
protenc proteins.fasta embeddings.lmdb --model_name=esm2_t33_650M_UR50D
```

### MCP server (for AI agents)

ProtEnc includes an MCP server so agents in Cursor, Claude Code, and other MCP clients can list models and embed protein sequences.

Install the package (see Development), then register the server in your MCP config:

```json
{
  "mcpServers": {
    "protenc": {
      "command": "protenc-mcp",
      "env": {
        "PROTENC_DEFAULT_MODEL": "esm2_t30",
        "PROTENC_DEVICE": "cuda"
      }
    }
  }
}
```

Available tools:

* `protenc_list_models` — list supported embedding models
* `protenc_get_model_info` — model family, embedding dimension, and layer count
* `protenc_embed_sequences` — embed one or more amino-acid sequences

Environment variables:

* `PROTENC_DEFAULT_MODEL` — default model alias (default: `esm2_t30`)
* `PROTENC_DEVICE` — torch device such as `cuda`, `cpu`, or `cuda:0`

The generated embeddings will be stored in a lmdb key-value store and can be easily accessed using the `read_from_lmdb` utility function:

```python
from protenc.utils import read_from_lmdb

for label, embed in read_from_lmdb('embeddings.lmdb'):
    print(label, embed)
```

**Features**

Input formats:
* CSV
* JSON
* [FASTA](https://en.wikipedia.org/wiki/FASTA_format)

Output formats:
* [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database)
* [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)

General:
* Multi-GPU inference via `torch.nn.DataParallel` (`--data_parallel`; ESMC models fall back to a single GPU)
* FP16 inference (`--amp`)

Development
-----------

Clone the repository:

```bash
git clone git+https://github.com/kklemon/protenc.git
```

Install dependencies via [Poetry](https://python-poetry.org/):

```bash
poetry install
```

Contribution
------------

Have feature ideas or found a bug? Love to see support for a new model? Feel free to [create an issue](https://github.com/kklemon/ProtEnc/issues/new).

Todo
----

- [ ] Support for more input formats
  - [X] CSV
  - [ ] Parquet
  - [X] FASTA
  - [X] JSON
- [ ] Support for more output formats
  - [X] LMDB
  - [X] HDF5
  - [ ] DataFrame
- [ ] Support for large models
  - [ ] Model offloading
  - [ ] Sharding
  - [ ] FlashAttention (via Kernl?)
- [ ] Support for more protein language models
  - [X] Whole ProtTrans family
  - [X] Whole ESM family
  - [ ] AlphaFold (?)
- [X] Implement all remaining TODOs in code
- [ ] Evaluation
- [ ] Demos
- [ ] Distributed inference
- [ ] Maybe support some sort of optimized inference such as quantization
  - This may be up to the model providers
- [ ] Improve documentation
- [ ] Support translation of gene sequences
- [ ] Add tests. We need tests!!!
