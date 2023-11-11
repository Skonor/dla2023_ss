import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file
from tqdm import tqdm


URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}

def _load_part(part, _data_dir):
    arch_path = _data_dir / f"{part}.tar.gz"
    print(f"Loading part {part}")
    download_file(URL_LINKS[part], arch_path)
    shutil.unpack_archive(arch_path, _data_dir)
    for fpath in (_data_dir / "LibriSpeech").iterdir():
        shutil.move(str(fpath), str(_data_dir / fpath.name))
    os.remove(str(arch_path))
    shutil.rmtree(str(_data_dir / "LibriSpeech"))

def main():
    data_dir = Path(__file__).absolute().resolve().parent.parent / "data" / "datasets" / "librispeech"
    for part in ['test-clean', 'train-clean-100']:
        part_dir = data_dir / part
        if not part_dir.exists():
            _load_part(part, data_dir)

if __name__ == "__main__":
    main()