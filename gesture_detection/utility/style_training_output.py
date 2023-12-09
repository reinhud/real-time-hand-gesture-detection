from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="purple4",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="purple4",
        metrics_text_delimiter="\n",
        metrics_format=".3e",
    )
)

model_summary = RichModelSummary(header_style="purple4")
