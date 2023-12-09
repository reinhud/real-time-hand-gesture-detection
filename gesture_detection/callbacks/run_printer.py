from lightning.pytorch.callbacks import Callback
from rich import get_console


class RunPrinter(Callback):
    def on_train_start(self, *args, **kwargs) -> None:
        console = get_console()
        console.rule("[purple]STARTED TRAINING LOOP", style="white")

    def on_train_end(self, *args, **kwargs) -> None:
        console = get_console()
        console.rule("[purple]FINISHED TRAINING LOOP", style="white")
