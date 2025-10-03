import pyarrow.parquet as pq
import pandas as pd

def process_one_file(filename, batch_size=50000, threshold=None):
    dataset = pq.ParquetFile(filename)
    results = []

    for batch in dataset.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()

        grouped = df.groupby(["input_idx", "modality", "layer", "neuron"])["value"].agg(
            max_activation="max",
            mean_activation="mean"
        )

        if threshold is not None:
            grouped["pct_above_thresh"] = (
                df.groupby(["input_idx", "modality", "layer", "neuron"])["value"]
                .apply(lambda x: (x > threshold).mean())
            )

        results.append(grouped)

    # Merge all chunks for this file
    summary = pd.concat(results).groupby(["input_idx", "modality", "layer", "neuron"]).max()

    return summary.reset_index()


all_summaries = []

for i in range(100):
    filename = f"/content/drive/MyDrive/activations_batch_{i}.parquet"
    summary = process_one_file(filename, batch_size=50000)
    all_summaries.append(summary)

big_summary = pd.concat(all_summaries, ignore_index=True)


big_summary.to_parquet("/content/drive/MyDrive/big_summary.parquet", index=False)

