import plotly.express as px

def plot_layer_hist(summary_df, metric="mean_diff"):
    for layer in summary_df["layer"].unique():
        layer_df = summary_df[summary_df["layer"] == layer]
        fig = px.histogram(
            layer_df,
            x=metric,
            nbins=50,
            title=f"Layer {layer} - Distribution of {metric}"
        )
        fig.show()

# Load summary (if not in memory)
summary_df = pd.read_parquet("/content/drive/MyDrive/neuron_summary.parquet")

# Plot distribution of neuron preferences
plot_layer_hist(summary_df, metric="mean_diff")  # can also use "t_value"
