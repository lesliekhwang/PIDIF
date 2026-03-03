import os
import ansys.fluent.core as pyfluent
from ansys.fluent.core.utils.fluent_version import FluentVersion


ANSYS_ROOT_V251 = "/usr/local/tools/ansys_inc/v251"


def _ensure_v251_env() -> None:
    # PyFluent looks up AWP_ROOT251 to locate Fluent v251 install
    os.environ.setdefault("AWP_ROOT251", ANSYS_ROOT_V251)


def launch_v251(
    nprocs: int = 4,
    mode: pyfluent.FluentMode = pyfluent.FluentMode.SOLVER,
    precision: str = "double",
):
    """
    Launch Fluent 2025R1 (v251) in headless mode.
    Use mode=FluentMode.MESHING for meshing workflow.
    """
    _ensure_v251_env()

    session = pyfluent.launch_fluent(
        product_version=FluentVersion.v251,
        mode=mode,
        processor_count=nprocs,
        ui_mode="no_gui",
        precision=precision,
    )
    return session


def safe_exit(session) -> None:
    """Exit Fluent session safely across session types/versions."""
    for m in ("exit", "close"):
        if hasattr(session, m):
            try:
                getattr(session, m)()
                return
            except Exception:
                pass
    # Last resort: do nothing (better than crashing your batch loop)
    return


if __name__ == "__main__":
    s = launch_v251(4, mode=pyfluent.FluentMode.SOLVER)
    print("Connected OK")
    safe_exit(s)