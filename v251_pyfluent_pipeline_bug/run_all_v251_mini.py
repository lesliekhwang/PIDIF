import csv
from pathlib import Path
from pipeline_mini import run_case_phase1

BASE_DIR = Path("/home/nuoxu9/PIDIF")
STEP_DIR = BASE_DIR / "step10"
RUN_DIR  = BASE_DIR / "runs_v251"

NPROCS = 4
NITER  = 200

def main():
    RUN_DIR.mkdir(exist_ok=True)

    with open(STEP_DIR / "designs.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    all_results = []

    for r in rows:
        case = r["case"]
        uin  = float(r["Uin_mps"])

        step_path = STEP_DIR / f"{case}.step"
        out_dir = RUN_DIR / case

        print(f"[RUN] {case}  Uin={uin}")

        res = run_case_phase1(
        out_dir=out_dir,
        uin=uin,
        nprocs=NPROCS,
        inlet_name="inlet",
        outlet_name="outlet",
        n_iter=NITER,
    )
    res["case"] = case
    all_results.append(res)

    # Write one summary CSV
    summary_path = RUN_DIR / "summary.csv"
    fieldnames = [
        "case", "uin", "n_iter",
        "p_in_Pa", "p_out_Pa", "dp_Pa",
        "mdot_in_kgps", "mdot_out_kgps",
        "mdot_in_abs_kgps", "mdot_out_abs_kgps",
        "inlet", "outlet",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in all_results:
            w.writerow({k: d.get(k, "") for k in fieldnames})

    print(f"[DONE] Wrote {summary_path}")

if __name__ == "__main__":
    main()