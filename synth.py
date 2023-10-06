from sdv.lite import SingleTablePreset
from sdv.datasets.local import load_csvs
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

datasets = load_csvs(".")
real_data = datasets["spotify_tracks"]

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

synthesizer = SingleTablePreset(
    metadata=metadata,
    name='FAST_ML'
)

synthesizer.fit(data=real_data)

synthetic_data = synthesizer.sample(num_rows=1000)
print(synthetic_data.head())

quality_report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

quality_report.get_visualization('Column Shapes').show()
