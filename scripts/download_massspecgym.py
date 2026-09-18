from pathlib import Path
from huggingface_hub import hf_hub_download

repo = "roman-bushuiev/MassSpecGym"
filename = "data/MassSpecGym1.5.tsv"
dest = Path("data")
dest.mkdir(exist_ok=True)
path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename)
out = dest / "MassSpecGym1.5.tsv"
if not out.exists():
    out.symlink_to(Path(path).resolve())
print(out)
