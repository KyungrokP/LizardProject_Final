from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _none_if_empty(value: str) -> str | None:
    v = value.strip()
    return v if v else None


def _none_if_negative(value: int) -> int | None:
    return value if value >= 0 else None


@app.command()
def main(
    dlc_config: str = typer.Option(..., help="Path to DeepLabCut config.yaml"),
    shuffle: int = typer.Option(1, min=1, help="Shuffle index"),
    trainingsetindex: int = typer.Option(0, min=0, help="Training set fraction index"),
    create_dataset: bool = typer.Option(
        True, help="Run create_training_dataset before training"
    ),
    train: bool = typer.Option(True, help="Run train_network"),
    evaluate: bool = typer.Option(True, help="Run evaluate_network after training"),
    plotting: bool = typer.Option(False, help="Enable evaluation plots"),
    engine: str = typer.Option(
        "",
        help="Optional DLC engine override (e.g., pytorch or tensorflow). "
        "Leave empty to use config.yaml value (recommended).",
    ),
    device: str = typer.Option(
        "cuda:0",
        help="Torch device string for DLC3 pytorch engine.",
    ),
    gputouse: str = typer.Option(
        "",
        help="GPU id for TensorFlow-style configs (optional).",
    ),
    batch_size: int = typer.Option(-1, help="Batch size override (-1 keeps config value)"),
    displayiters: int = typer.Option(
        -1, help="Print training status every N iterations (-1 disables override)"
    ),
    saveiters: int = typer.Option(
        -1, help="Save snapshot every N iterations (-1 disables override)"
    ),
    maxiters: int = typer.Option(
        -1, help="Stop training at iteration N (-1 disables override)"
    ),
    epochs: int = typer.Option(
        -1, help="Stop training at epoch N (PyTorch, -1 disables override)"
    ),
    save_epochs: int = typer.Option(
        -1, help="Save checkpoint every N epochs (PyTorch, -1 disables override)"
    ),
) -> None:
    cfg = Path(dlc_config)
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")

    try:
        import deeplabcut as dlc  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "DeepLabCut is not installed. Install optional deps: pip install -e '.[dlc]'"
        ) from exc

    engine_opt = _none_if_empty(engine)
    gpu_opt = _none_if_empty(gputouse)
    device_opt = _none_if_empty(device)

    if create_dataset:
        typer.echo("Creating DLC training dataset...")
        # NOTE: DLC 3.0.0rc13 can raise "Unknown augmentation for engine: pytorch"
        # when engine is explicitly passed here; rely on config.yaml engine instead.
        # Also allow reruns when a shuffle already exists.
        try:
            dlc.create_training_dataset(
                str(cfg),
                Shuffles=[shuffle],
                userfeedback=False,
            )
        except ValueError as exc:
            message = str(exc)
            if "already exists" in message and f"shuffle {shuffle}" in message:
                typer.echo(
                    f"Shuffle {shuffle} training dataset already exists; "
                    "skipping create_training_dataset."
                )
            else:
                raise

    if train:
        typer.echo("Training DLC network...")
        train_kwargs = dict(
            config=str(cfg),
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            displayiters=_none_if_negative(displayiters),
            saveiters=_none_if_negative(saveiters),
            maxiters=_none_if_negative(maxiters),
            epochs=_none_if_negative(epochs),
            save_epochs=_none_if_negative(save_epochs),
            gputouse=gpu_opt,
            batch_size=_none_if_negative(batch_size),
        )
        if device_opt is not None:
            train_kwargs["device"] = device_opt
        if engine_opt is not None:
            train_kwargs["engine"] = engine_opt
        dlc.train_network(**train_kwargs)

    if evaluate:
        typer.echo("Evaluating DLC network...")
        eval_kwargs = dict(
            config=str(cfg),
            Shuffles=[shuffle],
            trainingsetindex=trainingsetindex,
            plotting=plotting,
            gputouse=gpu_opt,
        )
        if engine_opt is not None:
            eval_kwargs["engine"] = engine_opt
        dlc.evaluate_network(**eval_kwargs)

    typer.echo("DLC training workflow complete.")


if __name__ == "__main__":
    app()
