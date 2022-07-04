import os

import click
from sklearn.datasets import load_wine


@click.command("download")
@click.option("--output-dir")
def download(output_dir: str):
    data, targets = load_wine(return_X_y=True, as_frame=True)
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    targets.to_csv(os.path.join(output_dir, "targets.csv"), index=False)


if __name__ == "__main__":
    download()
