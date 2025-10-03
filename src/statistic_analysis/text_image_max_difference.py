def modality_diff_restartable(big_summary_file, output_file, batch_size=50000, start_batch=0):
    dataset = pq.ParquetFile(big_summary_file)
    num_batches = dataset.num_row_groups  # how many row groups in the file

    # If file already exists, load it so we don’t redo previous work
    if os.path.exists(output_file):
        done = pd.read_parquet(output_file)
    else:
        done = pd.DataFrame(columns=["input_idx", "layer", "neuron", "modality_diff"])

    results = [done]

    for i in range(start_batch, num_batches):
        print(f"Processing batch {i+1}/{num_batches}...")

        batch = dataset.read_row_group(i).to_pandas()

        pivoted = batch.pivot_table(
            index=["input_idx", "layer", "neuron"],
            columns="modality",
            values="max_activation",
            aggfunc="max"
        ).reset_index()

        if "text" in pivoted.columns and "vision" in pivoted.columns:
            pivoted["modality_diff"] = pivoted["text"] - pivoted["vision"]
            results.append(pivoted[["input_idx", "layer", "neuron", "modality_diff"]])

        # Save progress after each batch
        partial = pd.concat(results).groupby(["input_idx", "layer", "neuron"]).max().reset_index()
        partial.to_parquet(output_file, index=False)

    print("Finished all batches.")
    return pd.read_parquet(output_file)

## full file process
modality_diff_restartable(big_summary_file="/content/drive/MyDrive/big_summary.parquet",
    output_file="/content/drive/MyDrive/modality_diff.parquet"
)
      
## partial process (if runtime disrupted)
modality_diff_restartable(
    big_summary_file="/content/drive/MyDrive/big_summary.parquet",
    output_file="/content/drive/MyDrive/modality_diff.parquet",
    start_batch=576   # resume at row group 71
)
