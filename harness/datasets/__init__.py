"""External evaluation sets (plan S0.7): sources, checksums and download.

Every file has a pinned URL (a commit, a fixed file id or a fixed object) and a
SHA-256 recorded when it was first downloaded (2026-11-13). Loaders verify the
checksum before parsing, so a silently changed upstream file fails loudly.

    python -m harness.datasets            # status of every dataset
    python -m harness.datasets fetch foil thumb ...

Only annotation files are fetched here. Images are listed in ``images`` for each set
and are fetched by the chat that needs them.

Access status:
  ``public``    True when the ORIGINAL source can be downloaded today without a form or
                login. False is a flag for the paper's reproducibility statement.
  ``fetchable`` True when every file has a pinned URL (an original or a mirror; ``access``
                says which). Pascal-50S is neither.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from harness import data_dir
from harness.download import ChecksumError, fetch, verify

GH = "https://raw.githubusercontent.com/{repo}/{rev}/{path}"
GDRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
THUMB_REV = "fd099b4f311d9c4fd24e6c58ef0df038ec1615f3"
VSR_REV = "b27a0af0ee1462d2b6b92c8c83e869d9254a241a"
AMBER_REV = "534babf6bbfcce2e735c26289dedfb21cef3c939"
WHATSUP_REV = "7c1f2550eace32e7b8c77de5a792347c402960d1"   # code that names the Drive ids
NOCAPS_MIRROR_REV = "50967f6f3616db2bf261e42b80377ab8cd8d4214"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str | None
    sha256: str | None
    size: int | None = None


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    homepage: str
    public: bool
    access: str
    images: str
    files: tuple[RemoteFile, ...]

    def root(self) -> Path:
        return data_dir() / "external" / self.key

    def path(self, name: str) -> Path:
        return self.root() / name

    def file(self, name: str) -> RemoteFile:
        return next(f for f in self.files if f.name == name)

    @property
    def fetchable(self) -> bool:
        """Every file has a pinned URL (possibly a mirror; see ``access``)."""
        return all(f.url for f in self.files)

    def local(self, name: str, verify_checksum: bool = True) -> Path:
        """Path of a downloaded file, checksum-verified. Raises if absent."""
        p = self.path(name)
        if not p.exists():
            hint = (f"run: python -m harness.datasets fetch {self.key}" if self.fetchable
                    else f"no download available: {self.access}")
            raise FileNotFoundError(f"{p} missing ({hint})")
        if verify_checksum:
            sha = self.file(name).sha256
            if sha is None:
                raise ChecksumError(f"{p}: no pinned sha256 yet for {self.key}/{name}")
            verify(p, sha)
        return p

    def fetch(self) -> list[Path]:
        if not self.fetchable:
            raise RuntimeError(f"{self.title}: {self.access}")
        return [fetch(f.url, self.path(f.name), f.sha256) for f in self.files]


def _gh(repo, rev, path, name, sha, size):
    return RemoteFile(name, GH.format(repo=repo, rev=rev, path=path), sha, size)


REGISTRY: dict[str, Dataset] = {d.key: d for d in [
    Dataset(
        "foil", "FOIL-COCO v1.0 (Oct 2018 release), test", "https://foilunitn.github.io/",
        True, "public Dropbox link from the project page",
        "COCO val2014 (official, real image ids): harness.coco.download_images",
        (RemoteFile("foilv1.0_test_2017.json",
                    "https://www.dropbox.com/s/u4ntgo73szg6yai/foilv1.0_test_2017.json?dl=1",
                    "7b37ab2a4e3dc911d006d5941919245c47d1e67d2ee29b4f7a87d5ecc9284642", 45080997),),
    ),
    Dataset(
        "thumb", "THumB 1.0 (MSCOCO)", "https://github.com/jungokasai/THumB",
        True, f"GitHub, commit {THUMB_REV[:7]} (repo has no license file)",
        "COCO val2014 (official, real image ids)",
        (_gh("jungokasai/THumB", THUMB_REV, "mscoco/mscoco_THumB-1.0.jsonl", "mscoco_THumB-1.0.jsonl",
             "463ebf947c793a541922ead33eb10a885e77c19e9c7d27cf89e034bfff643efa", 585297),
         _gh("jungokasai/THumB", THUMB_REV, "mscoco/mscoco_references.json", "mscoco_references.json",
             "897672f1f53ed33a376237c33fabc89ebbd04b40e9ced82691c2e9c9324d8fdf", 136692)),
    ),
    Dataset(
        "flickr8k", "Flickr8k-Expert and Flickr8k-CF (Hodosh et al., 2013)",
        "https://hockenmaier.cs.illinois.edu/8k-pictures.html",
        False,
        "original source is a sign-up form (forms.illinois.edu); files below are the "
        "CLIPScore authors' public Google Drive mirror (jmhessel/clipscore "
        "flickr8k_example/download.py), checksum-pinned",
        "Flickr8k_Dataset.zip, same mirror: Google Drive id 1WNY8pV-u8xtBYBVal03qwjQs4VKurUZn",
        (RemoteFile("Flickr8k_text.zip", GDRIVE.format("1ljB7DR-YM-q9WKnHDW5dHjauK029B2s6"),
                    "4992ddc8110e9aa49da5bf698522b0c8f11c448814a488584ee6bf040e5137e7", 2340801),),
    ),
    Dataset(
        "pascal50s", "Pascal-50S (Vedantam et al., 2015)", "http://ramavedantam.com/cider/",
        False,
        "the only official source (filebox.ece.vt.edu/~vrama91/CIDEr_miscellanous/"
        "pyCIDErConsensus.zip) does not respond and is not in the Wayback Machine; "
        "place pair_pascal.mat and consensus_pascal.mat in the dataset folder by hand "
        "and record their SHA-256 here",
        "PASCAL VOC 2010 JPEGImages (http://host.robots.ox.ac.uk/pascal/VOC/voc2010/)",
        (RemoteFile("pair_pascal.mat", None, None), RemoteFile("consensus_pascal.mat", None, None)),
    ),
    Dataset(
        "vsr", "Visual Spatial Reasoning (VSR)",
        "https://github.com/cambridgeltl/visual-spatial-reasoning",
        True, f"GitHub, commit {VSR_REV[:7]}, Apache-2.0",
        "COCO 2017 train/val images (official host; URL in each record's image_link)",
        tuple(_gh("cambridgeltl/visual-spatial-reasoning", VSR_REV, f"data/splits/{s}/{p}.jsonl",
                  f"{s}_{p}.jsonl", sha, size)
              for s, p, sha, size in [
                  ("random", "train", "8b725a8621f2d0f94745f1fefdcf3ff8a7eb96fd358676296ff9c9a4ff463c6e", 2362151),
                  ("random", "dev", "90de24b811597a913310d043758f1119a1ba9ebf1b1ca8f1e27c4907f01980d4", 338049),
                  ("random", "test", "8ade82a0b93ac9dc1e53f6cf1f11e9d5536776a3715102b6b4d27d4f81d551cc", 674712),
                  ("zeroshot", "train", "62daf2c5d573667038f88feaf1d26a1154c892083a72d83ed5c7551666ed3066", 1114884),
                  ("zeroshot", "dev", "0abda42fe9333ab6a8671b32886738b5cfa6d37a8c277b56d8368c339c8dffce", 107245),
                  ("zeroshot", "test", "914d9156723b912d8f794b179248af865a68c5f81c9d8e3a21d007778cacaad4", 380380),
              ]),
    ),
    Dataset(
        "whatsup", "What'sUp (Kamath et al., 2023)", "https://github.com/amitakamath/whatsup_vlms",
        True,
        f"public Google Drive files named in dataset_zoo/aro_datasets.py at commit "
        f"{WHATSUP_REV[:7]} (MIT)",
        "controlled_images / controlled_clevr: Drive ids 19KGYVQjrV3syb00GgcavB2nZTW5NXX0H, "
        "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG; COCO-QA uses COCO val2017; VG-QA uses Visual Genome",
        (RemoteFile("controlled_images_dataset.json", GDRIVE.format("1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm"),
                    "77ee94ae35552ed643ecd47f700daa178e10051458baae99d5c4c5d5922590d3", 88371),
         RemoteFile("controlled_clevr_dataset.json", GDRIVE.format("1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z"),
                    "2c0df1ab59d13d71442752b4b3c7353114d712a785b0850ec22e11c2b0c474c9", 86532),
         RemoteFile("coco_qa_one_obj.json", GDRIVE.format("1RsMdpE9mmwnK4zzMPpC1-wTU_hNis-dq"),
                    "5861b49d67d19482ea37b4b770a5cb59d7c08f156f71585cda20f324271e22f3", 182827),
         RemoteFile("coco_qa_two_obj.json", GDRIVE.format("1TCEoM0mgFmz8T4cF7PQ3XJmO6JjtiQ-s"),
                    "7dcdc1569565a9247a3e252a3d940b9600c4f17596652c947a461ff6ea13c57e", 42823),
         RemoteFile("vg_qa_one_obj.json", GDRIVE.format("1ARMRzRdohs9QTr1gpIfzyUzvW20wYp_p"),
                    "6f4af42c54a35902a4ce3884723d1d129dbd05921cd005512f1ded290bb7cb31", 96869),
         RemoteFile("vg_qa_two_obj.json", GDRIVE.format("1sjVG5O3QMY8s118k7kQM8zzDZH12i_95"),
                    "abeb1143db19a85aac678b9cd77a29381442f29caff475e2551a25abec2e4ecd", 31363)),
    ),
    Dataset(
        "amber", "AMBER (Wang et al., 2023)", "https://github.com/junyangwang0410/AMBER",
        True, f"GitHub, commit {AMBER_REV[:7]}, Apache-2.0",
        "AMBER images: public Google Drive id 1MaCHgtupcZUjf007anNl4_MV0o4DjXvl",
        tuple(_gh("junyangwang0410/AMBER", AMBER_REV, f"data/{p}", p.split("/")[-1], sha, size)
              for p, sha, size in [
                  ("annotations.json", "db22a436c4836fbb295e60b4e80d519573077579fb7032f01b193c1bd022b797", 1813301),
                  ("query/query_all.json", "127680ae61f7d4a92d2f7f07a8deaec07f0689b194d2c7c0f41ead7bd2c5bcf5", 1844750),
                  ("relation.json", "c4c1491e9bbc3209abc1888c12a1bf8413bdbab8a0d8be04b990a2e7680cf23e", 12699),
                  ("safe_words.txt", "f92cf1123f4578a2630832fac3e6ac4de36212ea55859d3bed4fd233723f4043", 57),
              ]),
    ),
    Dataset(
        "nocaps", "nocaps validation (Agrawal et al., 2019)", "https://nocaps.org/",
        False,
        "official S3 object (nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json) now "
        "returns 403; copy from the Hugging Face repo OpenGVLab/MM-Interleaved at commit "
        f"{NOCAPS_MIRROR_REV[:7]}, checksum-pinned (cannot be compared with the original; "
        "it parses to the published 4,500 images / 45,000 captions)",
        "Open Images V4 validation images by open_images_id (CVDF public bucket)",
        (RemoteFile("nocaps_val_4500_captions.json",
                    "https://huggingface.co/OpenGVLab/MM-Interleaved/resolve/"
                    f"{NOCAPS_MIRROR_REV}/assets/datasets/nocaps/nocaps_val_4500_captions.json",
                    "05e169ea5d9a8f3251e227d8900e5f3f7711d4b461cc397ed78312f0c917e386", 5934547),),
    ),
]}


def get(key: str) -> Dataset:
    return REGISTRY[key]


def status(key: str) -> str:
    d = REGISTRY[key]
    states = []
    for f in d.files:
        p = d.path(f.name)
        if not p.exists():
            states.append("missing")
            continue
        try:
            verify(p, f.sha256)
            states.append("ok" if f.sha256 else "unpinned")
        except ChecksumError:
            states.append("BAD CHECKSUM")
    flag = "" if d.public else "  [NOT PUBLIC: " + d.access + "]"
    return f"{key:10s} {','.join(sorted(set(states)))}{flag}"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m harness.datasets")
    sub = p.add_subparsers(dest="cmd")
    f = sub.add_parser("fetch")
    f.add_argument("keys", nargs="*", default=[k for k, d in REGISTRY.items() if d.fetchable])
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        for k in args.keys:
            for path in REGISTRY[k].fetch():
                print(f"ok {k}: {path}")
    for k in REGISTRY:
        print(status(k))


if __name__ == "__main__":
    main()
