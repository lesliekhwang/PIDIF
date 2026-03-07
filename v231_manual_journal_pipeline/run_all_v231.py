import subprocess
import csv
from pathlib import Path

BASE_DIR = Path("/home/nuoxu9/PIDIF")
MESH_DIR = BASE_DIR
RUN_DIR  = BASE_DIR / "runs"
TEMPLATE = BASE_DIR / "fluent_template.jou"

RUN_DIR.mkdir(exist_ok=True)

def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():

    with open(BASE_DIR / "step_out/designs.csv") as f:
        rows = list(csv.DictReader(f))

    template_text = TEMPLATE.read_text()

    for r in rows:

        case = r["case"]
        uin = float(r["Uin_mps"])

        mesh_file = BASE_DIR / f"{case}.msh"
        case_dir = RUN_DIR / case
        case_dir.mkdir(exist_ok=True)

        journal_text = template_text.format(
            MESH_FILE=str(mesh_file),
            UIN=uin,
            RUN_DIR=str(case_dir)
        )

        journal_path = case_dir / "run.jou"
        journal_path.write_text(journal_text)

        print(f"[RUN] {case}  Uin={uin}")

        run_cmd([
            "/usr/local/tools/ansys_inc/v232/fluent/bin/fluent",
            "2d",
            "-g",
            "-t4",
            "-i",
            str(journal_path)
        ])

if __name__ == "__main__":
    main()