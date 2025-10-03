import pandas as pd
from scipy import stats

def summarize_neurons(modality_diff_file, output_file=None):
    # Load modality difference file
    df = pd.read_parquet(modality_diff_file)

    summaries = []

    # Group by layer + neuron
    grouped = df.groupby(["layer", "neuron"])["modality_diff"]

    for (layer, neuron), diffs in grouped:
        diffs = diffs.dropna().values
        if len(diffs) > 1:  # need at least 2 samples
            t_val, p_val = stats.ttest_1samp(diffs, 0.0)
        else:
            t_val, p_val = None, None

        summaries.append({
            "layer": layer,
            "neuron": neuron,
            "mean_diff": diffs.mean(),
            "t_value": t_val,
            "p_value": p_val
        })

    summary_df = pd.DataFrame(summaries)

    if output_file:
        summary_df.to_parquet(output_file, index=False)

    return summary_df


summary_df = summarize_neurons(
    "/content/drive/MyDrive/modality_diff.parquet",
    output_file="/content/drive/MyDrive/neuron_summary.parquet"
)
