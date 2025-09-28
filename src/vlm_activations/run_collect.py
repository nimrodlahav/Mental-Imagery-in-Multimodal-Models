import argparse, os
from .model import load_model
from .data import load_batch
from .preprocess import text_inputs, image_inputs
from .collect import parquet_writer, collect_activations_one_pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--coco-root", required=True)
    ap.add_argument("--start-index", type=int, default=210)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--num-batches", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--outdir", default="./outputs")
    ap.add_argument("--quant-4bit", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    shuffled_ids_json = os.path.join(args.coco_root, "shuffled_ids.json")
    ann_file = os.path.join(args.coco_root, "annotations", "captions_val2017.json")

    # prepare shuffled ids once (idempotent)
    if not os.path.exists(shuffled_ids_json):
        from .data import shuffle_and_save_ids
        shuffle_and_save_ids(ann_file, shuffled_ids_json)

    model, processor = load_model(args.model_id, args.quant_4bit, "float16")
    device = next(model.parameters()).device
    layers = list(model.model.language_model.layers)

    for i in range(args.num-batches):
        global_idx = args.start_index//args.batch_size + i
        rows = load_batch(args.coco_root, shuffled_ids_json, args.start_index + i*args.batch_size, args.batch_size)
        fname = os.path.join(args.outdir, f"activations_batch_{global_idx}.parquet")
        with parquet_writer(fname) as writer:
            ti = text_inputs(rows, processor, device)
            collect_activations_one_pass(model, layers, ti, writer, global_idx, "text")
            ii = image_inputs(rows, processor, device)
            collect_activations_one_pass(model, layers, ii, writer, global_idx, "vision")

if __name__ == "__main__":
    main()
