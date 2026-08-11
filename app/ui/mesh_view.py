"""메쉬 → plotly Mesh3d 인자 변환 (streamlit 비의존)."""
from __future__ import annotations

import trimesh


def mesh_to_plotly(mesh: trimesh.Trimesh) -> dict:
    """정점 xyz + 삼각형 ijk — plotly.graph_objects.Mesh3d(**d)."""
    v = mesh.vertices
    f = mesh.faces
    return {
        "x": v[:, 0].tolist(), "y": v[:, 1].tolist(),
        "z": v[:, 2].tolist(),
        "i": f[:, 0].tolist(), "j": f[:, 1].tolist(),
        "k": f[:, 2].tolist(),
    }
