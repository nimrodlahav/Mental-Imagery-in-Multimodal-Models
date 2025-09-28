import pyarrow as pa
import pyarrow.parquet as pq

def forward(model, inputs, max_length=512):
    return model(**inputs, max_length=max_length, output_hidden_states=True, do_sample=False, return_dict=True)

def collect_activations_one_pass(model, layers, inputs, writer, global_index: int, typ: str):
    def make_hook(layer_idx, typ):
        def hook(module, input, output):
            layer_act = output.detach().cpu().numpy()
            bsz, seqlen, hdim = layer_act.shape
            entries = []
            for b in range(bsz):
                for t in range(seqlen):
                    for n_idx, val in enumerate(layer_act[b, t]):
                        entries.append({
                            "value": float(val),
                            "layer": layer_idx,
                            "batch": b,
                            "token": t,
                            "neuron": n_idx,
                            "input_idx": global_index * bsz + b,
                            "modality": typ
                        })
            table = pa.Table.from_pylist(entries, schema=writer.schema)
            writer.write_table(table)
        return hook

    hooks = [L.mlp.act_fn.register_forward_hook(make_hook(i, typ)) for i, L in enumerate(layers)]
    _ = forward(model, inputs)
    for h in hooks: h.remove()

def parquet_writer(path):
    schema = pa.schema([
        ("value", pa.float32()),
        ("layer", pa.int32()),
        ("batch", pa.int32()),
        ("token", pa.int32()),
        ("neuron", pa.int32()),
        ("input_idx", pa.int32()),
        ("modality", pa.string()),
    ])
    return pq.ParquetWriter(path, schema)
