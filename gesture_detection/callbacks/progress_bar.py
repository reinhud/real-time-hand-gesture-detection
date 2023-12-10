from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

theme = RichProgressBarTheme(
    description="purple4",
    progress_bar="red1",
    progress_bar_finished="green_yellow",
    progress_bar_pulse="#6206E0",
    batch_progress="purple4",
    time="grey82",
    processing_speed="grey82",
    metrics="purple4",
    metrics_text_delimiter="\n",
    metrics_format=".3e",
)


class CustomRichProgressBar(RichProgressBar):
    def __init__(self):
        super().__init__(theme=theme, leave=True)
