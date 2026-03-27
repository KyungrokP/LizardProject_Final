from __future__ import annotations

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    project_name: str = typer.Option(..., help="DeepLabCut project name"),
    experimenter: str = typer.Option(..., help="Experimenter name"),
    video: str = typer.Option(..., help="Sample video for project bootstrap"),
    working_directory: str = typer.Option(
        ".", help="Directory where the DLC project should be created"
    ),
) -> None:
    try:
        import deeplabcut  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "DeepLabCut is not installed. Install optional deps: pip install .[dlc]"
        ) from exc

    config_path = deeplabcut.create_new_project(
        project=project_name,
        experimenter=experimenter,
        videos=[video],
        working_directory=working_directory,
        copy_videos=False,
    )
    typer.echo(f"Created DeepLabCut project: {config_path}")


if __name__ == "__main__":
    app()

