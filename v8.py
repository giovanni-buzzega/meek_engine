# # Interactive Bayesian Network, V-Structure, Meek Rules & Gadget Explorer (`pyvis.network`)
# 
# This module provides a complete, extensible architecture (`BayesianNetworkExplorer` and `GadgetTemplate`) for interactively exploring **Bayesian network skeletons**, **reusable subgraph gadgets**, **v-structures (unshielded colliders)**, **no-v-structure constraints**, **Meek orientation rules ($R_1$–$R_4$) with cycle-avoidance lookahead**, and **inclusion-maximal v-structure DAG completion**.
# 
# ### Architectural Highlights
# 1. **Starting from an Undirected Skeleton or Reusable Gadgets (`GadgetTemplate`)**:
#    - Define any undirected graph $G = (V, E)$ or instantiate reusable **Gadgets** (e.g. a 6-node Pentagon + Apex subgraph or Variable & Glue gadgets with preset forbidden v-structures and a canonical geometric shape) via `add_gadget()`, `add_glue_gadget()`, `link_gadgets()`, and `remove_gadget()`.
#    - Move gadgets **as a rigid block** directly in the interactive `pyvis.network` canvas (dragging any node in a gadget translates all nodes and corner badges of that gadget together, with an interactive toggle to switch between rigid-block dragging and single-node dragging).
# 2. **Distinguishing When Incident Edges Can vs. Cannot Forbid a V-Structure**:
#    - **Shielded Triples (Triangles, $(u, v) \in E$)**: Marked as `SHIELDED_TRIANGLE` (`can_forbid=False`), because no v-structure can ever form when the endpoints $u$ and $v$ are adjacent.
#    - **Forced / Induced V-Structures (`ACTIVE_V_INDUCED`, `can_forbid=False`)**: On an **induced 4-cycle** $A - B - C - D - A$, forbidding v-structures on three corners $(D, A, B)$, $(A, B, C)$, and $(B, C, D)$ automatically induces $A \to D$ and $C \to D$ and marks $(C, D, A)$ as `ACTIVE_V_INDUCED` (`can_forbid=False`).
#    - **Induced Forbidden V-Structures (`INDUCED_FORBIDDEN_V`, `can_forbid=False`)**: Triples where neither edge is oriented yet, but orienting both $u \to w$ and $v \to w$ together would force a cycle under the current constraints.
#    - **Allowed / Open Triples (`ALLOWED_OPEN`, `can_forbid=True`)**: Unshielded pairs of incident edges where both forming a v-structure and avoiding a v-structure remain realizable in valid DAGs.
# 3. **Interactive Commands with Automatic Cycle Prevention & Culprit Witness Explanations**:
#    - `orient_edge(u, v)` & `forbid_v_structure(...)`: Propagate Meek closure ($R_1, R_2, R_3, R_4$) + lookahead cycle prevention. Any undirected edge where one direction would force a cycle under current constraints is **already oriented** in the opposite direction as an `INDUCED` edge, with an explicit shortest-path / culprit-node trace explaining why.
# 4. **Inclusion-Maximal V-Structure DAG Completion (`complete_maximal_v_structures`)**:
#    - Completely directs the partially directed graph into a valid DAG $D^*$ expanding on the v-structures already present such that the final set of v-structures $V(D^*)$ is **inclusion-maximal** ($\not\exists D'$ valid completion with $V(D^*) \subsetneq V(D')$).


"""
Bayesian Network Skeleton, Meek Rules, V-Structure & Gadget Interactive Explorer
================================================================================
Provides `GadgetTemplate` and `BayesianNetworkExplorer` for interactive exploration of:
  - Reusable subgraph gadgets with canonical shapes, namespaced IDs, preset constraints,
    clean insertion/removal, and rigid-block dragging in `pyvis.network`
  - Undirected skeleton graphs and incident edge pairs (triples)
  - User-chosen edge orientations & user-forbidden v-structures
  - Automatic Meek Rules (R1, R2, R3, R4) + Cycle-Lookahead & DAG-Backbone propagation
    with exact shortest-path and culprit-node witness explanations
  - Inclusion-maximal v-structure DAG completion (`complete_maximal_v_structures`)
"""

from __future__ import annotations

from collections import deque
import dataclasses
import html
import json
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

try:
    from pyvis.network import Network as PyVisNetwork
    _HAS_PYVIS = True
except ImportError:
    _HAS_PYVIS = False

try:
    from IPython.display import HTML, display
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False


Node = str
Edge = Tuple[Node, Node]            # Directed edge (u, v) meaning u -> v
UndirectedEdge = Tuple[Node, Node]  # Canonical undirected edge (min(u, v), max(u, v))
Triple = Tuple[Node, Node, Node]    # Canonical triple (u, w, v) with u < v, centered at w


def normalize_undirected(u: Node, v: Node) -> UndirectedEdge:
    u, v = str(u), str(v)
    return (u, v) if u < v else (v, u)


def normalize_triple(u: Node, w: Node, v: Node) -> Triple:
    u, w, v = str(u), str(w), str(v)
    if u == v or u == w or v == w:
        raise ValueError(f"Triple nodes must be distinct, got ({u}, {w}, {v}).")
    return (u, w, v) if u < v else (v, w, u)


def resolve_triple_args(*args: Any) -> Triple:
    """
    Accepts either:
      - Three node names: (u, w, v) where w is the center node, OR
      - A single 3-tuple of nodes: ((u, w, v),), OR
      - Two incident edges: ((u, w), (v, w)) in any vertex order, automatically
        identifying the shared vertex w as the center of the triple.
    """
    if len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 3:
        u, w, v = args[0]
        return normalize_triple(u, w, v)
    if len(args) == 3:
        return normalize_triple(args[0], args[1], args[2])
    if len(args) == 2 and all(isinstance(a, (tuple, list)) and len(a) == 2 for a in args):
        e1 = (str(args[0][0]), str(args[0][1]))
        e2 = (str(args[1][0]), str(args[1][1]))
        shared = set(e1) & set(e2)
        if len(shared) != 1:
            raise ValueError(
                f"Edge pair {e1} and {e2} must share exactly one common endpoint to form an incident pair."
            )
        w = next(iter(shared))
        u = e1[0] if e1[1] == w else e1[1]
        v = e2[0] if e2[1] == w else e2[1]
        return normalize_triple(u, w, v)
    raise ValueError(
        "Specify a triple either as three nodes `(u, w, v)` (with center `w`) "
        "or as two incident edges `((u, w), (v, w))`."
    )


# ---------------------------------------------------------------------------
# Reusable Gadget Template & Instance Definitions
# ---------------------------------------------------------------------------

GADGET_PALETTE = [
    {"border": "#1d4ed8", "bg": "#eff6ff", "hull": "rgba(59, 130, 246, 0.08)"},
    {"border": "#0f766e", "bg": "#f0fdfa", "hull": "rgba(20, 184, 166, 0.08)"},
    {"border": "#b45309", "bg": "#fffbeb", "hull": "rgba(245, 158, 11, 0.08)"},
    {"border": "#6d28d9", "bg": "#f5f3ff", "hull": "rgba(139, 92, 246, 0.08)"},
    {"border": "#be185d", "bg": "#fdf2f8", "hull": "rgba(236, 72, 153, 0.08)"},
]


@dataclasses.dataclass
class GadgetTemplate:
    """
    Reusable blueprint for a subgraph "gadget" with:
      - `name`: human-readable template name (e.g. "PentagonApex")
      - `local_nodes`: list of local node role names (e.g. ["p1", "p2", "p3", "p4", "p5", "apex"])
      - `local_edges`: list of undirected skeleton edges between local nodes
      - `local_positions`: dict mapping each local node to canonical relative (dx, dy) coordinates
        around (0, 0) so every instance has the exact same recognizable geometric shape
      - `forbidden_v`: list of local triples (u, w, v) or incident edge pairs where v-structures
        are forbidden by default
      - `oriented_edges`: optional list of preset directed edges (u, v) inside the gadget
    """
    name: str
    local_nodes: List[Node]
    local_edges: List[Tuple[Node, Node]]
    local_positions: Dict[Node, Tuple[float, float]]
    forbidden_v: List[Any] = dataclasses.field(default_factory=list)
    oriented_edges: List[Tuple[Node, Node]] = dataclasses.field(default_factory=list)

    @classmethod
    def pentagon_with_apex(
        cls,
        radius: float = 95.0,
        apex_distance: float = 165.0,
        forbidden_v: Optional[List[Any]] = None,
        apex_links: Tuple[Node, ...] = ("p1",),
    ) -> "GadgetTemplate":
        """
        Factory for a 6-node gadget consisting of a regular 5-cycle (pentagon p1..p5)
        plus a 6th node (`apex`) connected to `apex_links` (default: `("p1",)`),
        with customizable preset forbidden v-structures.
        """
        ring = ["p1", "p2", "p3", "p4", "p5"]
        local_nodes = ring + ["apex"]
        local_edges: List[Tuple[Node, Node]] = [
            ("p1", "p2"), ("p2", "p3"), ("p3", "p4"), ("p4", "p5"), ("p5", "p1")
        ]
        for target in apex_links:
            local_edges.append((target, "apex"))

        # Standard recognizable geometry: regular pentagon with p1 at the top (0, -radius)
        # and `apex` extending straight above p1 at (0, -apex_distance)
        pos: Dict[Node, Tuple[float, float]] = {}
        for i, n in enumerate(ring):
            angle = -math.pi / 2 + (2 * math.pi * i) / 5
            pos[n] = (radius * math.cos(angle), radius * math.sin(angle))
        pos["apex"] = (0.0, -apex_distance)

        default_forbidden = (
            forbidden_v
            if forbidden_v is not None
            else [
                ("p5", "p1", "p2"),
                ("p1", "p2", "p3"),
                ("p3", "p4", "p5"),
                ("apex", "p1", "p2"),
            ]
        )
        return cls(
            name="PentagonApex6",
            local_nodes=local_nodes,
            local_edges=local_edges,
            local_positions=pos,
            forbidden_v=default_forbidden,
        )

    @classmethod
    def variable_gadget(
        cls,
        radius: float = 150.0,
        include_o: bool = False,
    ) -> "GadgetTemplate":
        """
        5-node variable gadget ring `["l", "tl", "tr", "br", "n"]` with its ring-internal
        forbidden v-structures `[("n", "br", "tr"), ("tl", "l", "n"), ("l", "tl", "tr")]`.
        By default (`include_o=False`), `"o"` nodes are dynamically injected into each
        variable gadget by `explorer.add_glue_gadget(...)` (one `"o"` node per glue gadget),
        which also enforces the `"o"`-specific forbidden v-structures `("o", "n", "br")`
        and `("tl", "o", "n")`. Set `include_o=True` to include a standalone `"o"` node.
        """
        ring = ["l", "tl", "tr", "br", "n"]
        local_nodes = list(ring) + (["o"] if include_o else [])
        local_edges: List[Tuple[Node, Node]] = [
            ("l", "tl"), ("tl", "tr"), ("tr", "br"), ("br", "n"), ("n", "l"),
        ]
        if include_o:
            local_edges.extend([("n", "o"), ("tl", "o"), ("l", "o")])

        pos: Dict[Node, Tuple[float, float]] = {}
        for i, n in enumerate(ring):
            angle = -math.pi + (2 * math.pi * i) / 5
            pos[n] = (radius * math.cos(angle), radius * math.sin(angle))
        if include_o:
            pos["o"] = (0.0, 0.0)

        default_forbidden: List[Any] = [
            ("n", "br", "tr"),
            ("tl", "l", "n"),
            ("l", "tl", "tr"),
        ]
        if include_o:
            default_forbidden.extend([
                ("o", "n", "br"),
                ("tl", "o", "n"),
            ])

        return cls(
            name="gadgetVar",
            local_nodes=local_nodes,
            local_edges=local_edges,
            local_positions=pos,
            forbidden_v=default_forbidden,
        )


@dataclasses.dataclass
class GadgetInstance:
    """Tracks an instantiated gadget inside `BayesianNetworkExplorer`."""
    instance_id: str
    template: GadgetTemplate
    center: Tuple[float, float]
    scale: float
    angle_deg: float
    palette_idx: int
    node_map: Dict[Node, Node]  # local_name -> global_name (e.g. "l" -> "G1.l")
    local_positions: Dict[Node, Tuple[float, float]] = dataclasses.field(default_factory=dict)
    glue_o_nodes: Dict[str, Node] = dataclasses.field(default_factory=dict)  # glue_id -> global o_node

    def node(self, local_name: Node) -> Node:
        """Return the full graph node ID for a local gadget role (e.g. `g.node('tl') -> 'G1.tl'`)."""
        return self.node_map[local_name]


class _FallbackPyVisNetwork:
    """
    Drop-in fallback matching `pyvis.network.Network` API when `pyvis` is not installed
    in the current Python environment, generating standalone `vis-network` HTML.
    """
    def __init__(self, height: str = "600px", width: str = "100%", directed: bool = True,
                 notebook: bool = True, bgcolor: str = "#ffffff", font_color: str = "#0f172a",
                 cdn_resources: str = "remote", **kwargs: Any):
        self.height = height
        self.width = width
        self.directed = directed
        self.notebook = notebook
        self.bgcolor = bgcolor
        self.font_color = font_color
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.options: Dict[str, Any] = {}

    def add_node(self, n_id: Union[str, int], label: Optional[str] = None, **kwargs: Any) -> None:
        node = {"id": str(n_id), "label": str(label if label is not None else n_id)}
        node.update(kwargs)
        self.nodes.append(node)

    def add_edge(self, source: Union[str, int], to: Union[str, int], **kwargs: Any) -> None:
        edge = {"from": str(source), "to": str(to)}
        edge.update(kwargs)
        self.edges.append(edge)

    def set_options(self, options_json: str) -> None:
        self.options = json.loads(options_json)

    def generate_html(self, name: str = "index.html", notebook: bool = True) -> str:
        uid = f"vis_net_{random.randint(10000, 99999)}"
        nodes_json = json.dumps(self.nodes)
        edges_json = json.dumps(self.edges)
        opts_json = json.dumps(self.options)
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body style="margin:0;padding:0;background:{self.bgcolor};">
<div id="{uid}" style="width:{self.width};height:{self.height};border:1px solid #e2e8f0;border-radius:8px;background:{self.bgcolor};"></div>
<script type="text/javascript">
(function() {{
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});
  var container = document.getElementById("{uid}");
  var data = {{ nodes: nodes, edges: edges }};
  var options = {opts_json};
  var network = new vis.Network(container, data, options);
  window["__bn_vis_" + "{uid}"] = {{ network: network, nodes: nodes, edges: edges }};
}})();
</script>
</body>
</html>"""

    def save_graph(self, name: str) -> None:
        with open(name, "w", encoding="utf-8") as f:
            f.write(self.generate_html(name=name))


# Visual palette constants
EDGE_STYLES = {
    "UNDIRECTED": {
        "color": "#64748b",
        "width": 3.0,
        "dashes": False,
        "arrows": "",
        "label_tag": "Undirected (Freely Directable)",
    },
    "USER": {
        "color": "#2563eb",
        "width": 4.5,
        "dashes": False,
        "arrows": "to",
        "label_tag": "User-Chosen Orientation",
    },
    "INDUCED": {
        "color": "#d97706",
        "width": 4.0,
        "dashes": [8, 5],
        "arrows": "to",
        "label_tag": "Induced Orientation (Meek / Cycle Avoidance)",
    },
    "COMPLETION_V": {
        "color": "#7c3aed",
        "width": 4.0,
        "dashes": [4, 4],
        "arrows": "to",
        "label_tag": "Maximal Completion (New V-Structure Edge)",
    },
    "COMPLETION_INDUCED": {
        "color": "#0d9488",
        "width": 3.5,
        "dashes": [6, 4],
        "arrows": "to",
        "label_tag": "Maximal Completion (Meek-Induced / DAG Fill)",
    },
}

TRIPLE_STYLES = {
    "ALLOWED_OPEN": {
        "color": "#0284c7",
        "bg": "#e0f2fe",
        "shape": "dot",
        "label": "Allowed V (Can Forbid)",
    },
    "FORBIDDEN_USER": {
        "color": "#dc2626",
        "bg": "#fee2e2",
        "shape": "box",
        "label": "Forbidden V (User Constraint)",
    },
    "INDUCED_FORBIDDEN_V": {
        "color": "#e11d48",
        "bg": "#ffe4e6",
        "shape": "box",
        "label": "Induced Forbidden V (Cycle Constraint)",
    },
    "ACTIVE_V_USER": {
        "color": "#16a34a",
        "bg": "#dcfce7",
        "shape": "diamond",
        "label": "Active V-Structure (User)",
    },
    "ACTIVE_V_INDUCED": {
        "color": "#ea580c",
        "bg": "#ffedd5",
        "shape": "diamond",
        "label": "Forced/Induced V-Structure (Cannot Forbid)",
    },
    "ACTIVE_V_COMPLETION": {
        "color": "#7c3aed",
        "bg": "#ede9fe",
        "shape": "diamond",
        "label": "Active V-Structure (Maximal Completion)",
    },
    "PREVENTED_NON_COLLIDER": {
        "color": "#64748b",
        "bg": "#f1f5f9",
        "shape": "ellipse",
        "label": "Cannot Forbid (Already Non-Collider)",
    },
    "SHIELDED_TRIANGLE": {
        "color": "#94a3b8",
        "bg": "#f8fafc",
        "shape": "triangle",
        "label": "Cannot Forbid (Shielded / Triangle)",
    },
}

# Categories whose corner annotations are hidden by default on the canvas
# (and whose toggle checkboxes start unchecked by default).
DEFAULT_HIDDEN_TRIPLE_CATEGORIES: Set[str] = {
    "ALLOWED_OPEN",
    "FORBIDDEN_USER",
    "INDUCED_FORBIDDEN_V",
    "ACTIVE_V_USER",
    "ACTIVE_V_INDUCED",
    "ACTIVE_V_COMPLETION",
    "PREVENTED_NON_COLLIDER",
    "SHIELDED_TRIANGLE",
}


class BayesianNetworkExplorer:
    """
    Interactive engine and PyVis visualizer for exploring Bayesian network skeletons,
    reusable subgraph gadgets, v-structures, no-v-structure constraints, Meek rules,
    cycle prevention, and inclusion-maximal v-structure DAG completions.
    """

    def __init__(
        self,
        nodes: Optional[Iterable[Node]] = None,
        edges: Optional[Iterable[Tuple[Node, Node]]] = None,
    ):
        self.nodes: Set[Node] = {str(n) for n in nodes} if nodes else set()
        self.skeleton: Set[UndirectedEdge] = set()
        self.adj: Dict[Node, Set[Node]] = {n: set() for n in self.nodes}

        # Custom node positions & gadget registry
        self.custom_positions: Dict[Node, Tuple[float, float]] = {}
        self.node_gadget: Dict[Node, str] = {}
        self.gadgets: Dict[str, GadgetInstance] = {}
        self.glue_gadgets: Dict[str, Dict[str, Any]] = {}
        self._witness_dags: List[Set[Edge]] = []

        # User-specified constraints
        self.user_oriented: Dict[UndirectedEdge, Edge] = {}
        self.user_forbidden_v: Set[Triple] = set()

        # Induced / derived state (computed automatically by _propagate)
        self.induced_oriented: Dict[UndirectedEdge, Edge] = {}
        self.induced_reasons: Dict[UndirectedEdge, str] = {}
        self.induced_rule_codes: Dict[UndirectedEdge, str] = {}

        # Maximal completion state (populated when complete_maximal_v_structures is called)
        self.completion_oriented: Dict[UndirectedEdge, Edge] = {}
        self.completion_edge_roles: Dict[UndirectedEdge, str] = {}
        self.propagation_log: List[str] = []

        if edges:
            for u, v in edges:
                self.add_skeleton_edge(u, v, propagate=False)
        self._propagate()

    # ------------------------------------------------------------------
    # Skeleton Construction, Node/Edge Removal & Reset
    # ------------------------------------------------------------------
    def add_node(self, u: Node, pos: Optional[Tuple[float, float]] = None) -> None:
        u = str(u)
        if u not in self.nodes:
            self.nodes.add(u)
            self.adj[u] = set()
        if pos is not None:
            self.custom_positions[u] = (float(pos[0]), float(pos[1]))

    def add_skeleton_edge(self, u: Node, v: Node, propagate: bool = True) -> None:
        u, v = str(u), str(v)
        if u == v:
            raise ValueError("Self-loops are not allowed in a Bayesian network skeleton.")
        self.add_node(u)
        self.add_node(v)
        e = normalize_undirected(u, v)
        self.skeleton.add(e)
        self.adj[u].add(v)
        self.adj[v].add(u)
        # If adding this chord turns an unshielded triple into a triangle, remove any forbidden_v on it
        shielded_now = {t for t in self.user_forbidden_v if normalize_undirected(t[0], t[2]) == e}
        self.user_forbidden_v -= shielded_now
        if propagate:
            self._propagate()

    def remove_skeleton_edge(self, u: Node, v: Node, propagate: bool = True) -> None:
        """Remove an undirected edge (and any orientation or dependent triple constraints) from the skeleton."""
        e = normalize_undirected(str(u), str(v))
        if e not in self.skeleton:
            raise ValueError(f"Edge {e} is not in the skeleton.")
        self.skeleton.remove(e)
        self.adj[e[0]].discard(e[1])
        self.adj[e[1]].discard(e[0])
        self.user_oriented.pop(e, None)
        self.completion_oriented.clear()
        self.completion_edge_roles.clear()
        # Remove any forbidden v-structures that used edge e
        self.user_forbidden_v = {
            t for t in self.user_forbidden_v
            if normalize_undirected(t[0], t[1]) != e and normalize_undirected(t[1], t[2]) != e
        }
        if propagate:
            self._propagate()

    def remove_node(self, u: Node, propagate: bool = True) -> None:
        """Remove a node and all its incident edges and constraints from the graph."""
        u = str(u)
        if u not in self.nodes:
            raise ValueError(f"Node '{u}' is not in the graph.")
        for nbr in list(self.adj[u]):
            self.remove_skeleton_edge(u, nbr, propagate=False)
        self.nodes.remove(u)
        del self.adj[u]
        self.custom_positions.pop(u, None)
        self.node_gadget.pop(u, None)
        self.user_forbidden_v = {t for t in self.user_forbidden_v if u not in t}
        if propagate:
            self._propagate()

    # ------------------------------------------------------------------
    # Gadget Management: Add, Move/Rotate, Link & Remove Gadgets
    # ------------------------------------------------------------------
    @staticmethod
    def _transform_point(
        dx: float, dy: float, center: Tuple[float, float], scale: float, angle_deg: float
    ) -> Tuple[float, float]:
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rx = scale * (dx * cos_a - dy * sin_a)
        ry = scale * (dx * sin_a + dy * cos_a)
        return (center[0] + rx, center[1] + ry)

    def add_gadget(
        self,
        template: GadgetTemplate,
        instance_id: str,
        center: Optional[Tuple[float, float]] = None,
        scale: float = 1.0,
        angle_deg: float = 0.0,
        apply_preset_constraints: bool = True,
        separator: str = ".",
    ) -> GadgetInstance:
        """
        Instantiate a `GadgetTemplate` into the graph under namespace `instance_id`
        (e.g. `instance_id="G1"` creates nodes `"G1.p1"`, `"G1.apex"`, etc.).
        Automatically places nodes in the template's canonical geometric shape around `center`
        and applies the template's preset forbidden v-structures and edge orientations.
        """
        instance_id = str(instance_id)
        if instance_id in self.gadgets:
            raise ValueError(f"Gadget instance '{instance_id}' already exists.")

        if center is None:
            # Automatically offset each new gadget horizontally if center is not provided
            idx = len(self.gadgets)
            center = (idx * 380.0, 0.0)

        palette_idx = len(self.gadgets) % len(GADGET_PALETTE)
        node_map: Dict[Node, Node] = {}

        for local_n in template.local_nodes:
            global_n = f"{instance_id}{separator}{local_n}"
            if global_n in self.nodes:
                raise ValueError(f"Node '{global_n}' already exists in the graph.")
            node_map[local_n] = global_n

        # 1. Add nodes with transformed canonical coordinates
        for local_n, global_n in node_map.items():
            dx, dy = template.local_positions.get(local_n, (0.0, 0.0))
            pos = self._transform_point(dx, dy, center, scale, angle_deg)
            self.add_node(global_n, pos=pos)
            self.node_gadget[global_n] = instance_id

        # 2. Add internal skeleton edges
        for u_loc, v_loc in template.local_edges:
            self.add_skeleton_edge(node_map[u_loc], node_map[v_loc], propagate=False)

        # 3. Apply preset forbidden v-structures and orientations
        if apply_preset_constraints:
            for item in template.forbidden_v:
                if isinstance(item, (tuple, list)) and len(item) == 3 and all(isinstance(x, str) for x in item):
                    t_global = normalize_triple(node_map[item[0]], node_map[item[1]], node_map[item[2]])
                else:
                    loc_t = resolve_triple_args(*item)
                    t_global = normalize_triple(node_map[loc_t[0]], node_map[loc_t[1]], node_map[loc_t[2]])
                if not self.has_edge(t_global[0], t_global[2]):
                    self.user_forbidden_v.add(t_global)

            for u_loc, v_loc in template.oriented_edges:
                gu, gv = node_map[u_loc], node_map[v_loc]
                self.user_oriented[normalize_undirected(gu, gv)] = (gu, gv)

        self._propagate()
        inst = GadgetInstance(
            instance_id=instance_id,
            template=template,
            center=(float(center[0]), float(center[1])),
            scale=scale,
            angle_deg=angle_deg,
            palette_idx=palette_idx,
            node_map=node_map,
            local_positions=dict(template.local_positions),
        )
        self.gadgets[instance_id] = inst
        return inst

    def move_gadget(
        self,
        instance_id: str,
        center: Optional[Tuple[float, float]] = None,
        scale: Optional[float] = None,
        angle_deg: Optional[float] = None,
    ) -> GadgetInstance:
        """
        Translate, rescale, or rotate an entire gadget instance as a rigid block in Python.
        (In the interactive PyVis visualization, you can also drag any node of a gadget
        to move the entire gadget block in real time!)
        """
        if instance_id not in self.gadgets:
            raise ValueError(f"Unknown gadget instance '{instance_id}'.")
        inst = self.gadgets[instance_id]
        if center is not None:
            inst.center = (float(center[0]), float(center[1]))
        if scale is not None:
            inst.scale = float(scale)
        if angle_deg is not None:
            inst.angle_deg = float(angle_deg)

        for local_n, global_n in inst.node_map.items():
            dx, dy = inst.local_positions.get(local_n, inst.template.local_positions.get(local_n, (0.0, 0.0)))
            self.custom_positions[global_n] = self._transform_point(
                dx, dy, inst.center, inst.scale, inst.angle_deg
            )
        return inst

    def add_glue_gadget(
        self,
        glue_id: str,
        var_gadgets: Iterable[Union[str, GadgetInstance]],
        separator: str = ".",
    ) -> Dict[str, Any]:
        """
        Glue an ordered sequence of k >= 2 distinct variable gadgets (G_1, G_2, ..., G_k):
          1. Adds an `"o"` node `o_i` inside each `G_i` (named `{G_i}.o_{glue_id}`, or `{G_i}.o`
             if `glue_id == "o"`), connected to `G_i`'s ring via `(n_i, o_i)`, `(tl_i, o_i)`, `(l_i, o_i)`.
          2. Enforces the `"o"`-specific internal forbidden v-structures on each `G_i`:
             `(o_i, n_i, br_i)` and `(tl_i, o_i, n_i)`.
          3. Connects `{o_1, ..., o_k}` in a clique: `(o_i, o_j)` for all i != j.
          4. Adds cross-gadget edges `(tl_i, o_j)`, `(l_i, o_j)`, `(tl_i, tl_j)`, `(tl_i, l_j)`,
             and `(l_i, l_j)` for all j != i.
          5. Forbids the v-structures `(n_i, o_i, o_{(i+1) mod k})` for every i = 0 .. k-1.
        """
        glue_id = str(glue_id)
        if glue_id in self.glue_gadgets:
            raise ValueError(f"Glue gadget '{glue_id}' already exists.")

        instances: List[GadgetInstance] = []
        seen_ids: Set[str] = set()
        for item in var_gadgets:
            inst = item if isinstance(item, GadgetInstance) else self.gadgets.get(str(item))
            if inst is None or inst.instance_id not in self.gadgets:
                raise ValueError(f"Variable gadget '{item}' is not registered in this explorer.")
            if inst.instance_id in seen_ids:
                raise ValueError(f"Duplicate variable gadget '{inst.instance_id}' in add_glue_gadget; all must be distinct.")
            for req_role in ("l", "tl", "tr", "br", "n"):
                if req_role not in inst.node_map:
                    raise ValueError(
                        f"Gadget '{inst.instance_id}' is missing required variable-gadget role '{req_role}'."
                    )
            seen_ids.add(inst.instance_id)
            instances.append(inst)

        k = len(instances)
        if k < 2:
            raise ValueError(f"add_glue_gadget requires at least 2 distinct variable gadgets, got {k}.")

        if self.completion_oriented:
            self.clear_completion()

        o_nodes: List[Node] = []
        local_o_names: List[str] = []

        # Step 1 & 2: Create o_i in each G_i, add internal edges & internal o-forbidden v-structures
        for inst in instances:
            local_o = "o" if (glue_id == "o" and "o" not in inst.node_map) else f"o_{glue_id}"
            global_o = f"{inst.instance_id}{separator}{local_o}"
            if global_o in self.nodes:
                raise ValueError(f"Node '{global_o}' already exists.")

            # Position first 'o' node at (0, 0) of G_i; offset additional 'o' nodes on a small inner circle
            num_existing_o = len(inst.glue_o_nodes)
            if num_existing_o == 0:
                dx, dy = (0.0, 0.0)
            else:
                ang = -math.pi / 2 + (2 * math.pi * num_existing_o) / 6.0
                dx, dy = (34.0 * math.cos(ang), 34.0 * math.sin(ang))

            inst.local_positions[local_o] = (dx, dy)
            inst.node_map[local_o] = global_o
            inst.glue_o_nodes[glue_id] = global_o

            pos = self._transform_point(dx, dy, inst.center, inst.scale, inst.angle_deg)
            self.add_node(global_o, pos=pos)
            self.node_gadget[global_o] = inst.instance_id
            o_nodes.append(global_o)
            local_o_names.append(local_o)

            # Internal edges: (n_i, o_i), (tl_i, o_i), (l_i, o_i)
            n_i = inst.node("n")
            tl_i = inst.node("tl")
            l_i = inst.node("l")
            br_i = inst.node("br")
            self.add_skeleton_edge(n_i, global_o, propagate=False)
            self.add_skeleton_edge(tl_i, global_o, propagate=False)
            self.add_skeleton_edge(l_i, global_o, propagate=False)

            # Internal 'o'-specific forbidden v-structures: ("o", "n", "br") and ("tl", "o", "n")
            self.user_forbidden_v.add(normalize_triple(global_o, n_i, br_i))
            self.user_forbidden_v.add(normalize_triple(tl_i, global_o, n_i))

        # Step 3 & 4: Clique on {o_1, ..., o_k} and cross edges (tl_i, o_j), (l_i, o_j) for j != i
        # UPDATED, add cross edge (tl_i, tl_j), (tl_i, l_j), (l_i, l_j) for j != i, and forbid v-structures instead
        for i in range(k):
            inst_i = instances[i]
            tl_i = inst_i.node("tl")
            l_i = inst_i.node("l")
            for j in range(k):
                if i < j:
                    self.add_skeleton_edge(o_nodes[i], o_nodes[j], propagate=False)
                if i != j:
                    self.add_skeleton_edge(tl_i, o_nodes[j], propagate=False)
                    self.add_skeleton_edge(l_i, o_nodes[j], propagate=False)
                    self.add_skeleton_edge(tl_i, instances[j].node("tl"), propagate=False)
                    self.add_skeleton_edge(tl_i, instances[j].node("l"), propagate=False)
                    self.add_skeleton_edge(l_i, instances[j].node("l"), propagate=False)

        # Step 5: Forbidden v-structures (n_i, o_i, o_{(i+1) mod k})
        for i in range(k):
            inst_i = instances[i]
            n_i = inst_i.node("n")
            o_i = o_nodes[i]
            next_o = o_nodes[(i + 1) % k]
            t_noo = normalize_triple(n_i, o_i, next_o)
            if not self.has_edge(t_noo[0], t_noo[2]):
                self.user_forbidden_v.add(t_noo)

        prev_induced = set(self.induced_oriented.values())
        self._propagate()
        newly_induced = [
            f"{src} -> {dst} [{self.induced_reasons[normalize_undirected(src, dst)]}]"
            for src, dst in sorted(set(self.induced_oriented.values()) - prev_induced)
        ]

        self.glue_gadgets[glue_id] = {
            "glue_id": glue_id,
            "gadget_ids": [inst.instance_id for inst in instances],
            "o_nodes": o_nodes,
            "local_o_names": local_o_names,
        }
        return {
            "status": True,
            "glue_id": glue_id,
            "o_nodes": o_nodes,
            "new_induced": newly_induced,
            "message": (
                f"Added glue gadget '{glue_id}' across {[inst.instance_id for inst in instances]} "
                f"(created {o_nodes}, induced {len(newly_induced)} edge(s))."
            ),
        }

    def remove_glue_gadget(self, glue_id: str) -> Dict[str, Any]:
        """Remove a glue gadget and all `'o'` nodes, edges, and constraints it introduced."""
        if glue_id not in self.glue_gadgets:
            raise ValueError(f"Unknown glue gadget '{glue_id}'.")
        info = self.glue_gadgets.pop(glue_id)
        for gid, local_o, global_o in zip(info["gadget_ids"], info["local_o_names"], info["o_nodes"]):
            if gid in self.gadgets:
                inst = self.gadgets[gid]
                inst.node_map.pop(local_o, None)
                inst.local_positions.pop(local_o, None)
                inst.glue_o_nodes.pop(glue_id, None)
            if global_o in self.nodes:
                self.remove_node(global_o, propagate=False)
        self._propagate()
        return {
            "status": True,
            "removed_glue_id": glue_id,
            "removed_o_nodes": info["o_nodes"],
            "message": f"Removed glue gadget '{glue_id}' and its o-nodes {info['o_nodes']}.",
        }

    def link_gadgets(
        self,
        node_a: Union[Node, Tuple[str, Node]],
        node_b: Union[Node, Tuple[str, Node]],
        oriented: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Link two gadgets (or any two nodes) with a skeleton edge, and optionally orient it.
        Nodes can be passed either as `"G1.apex"` or as a tuple `("G1", "apex")`.
        `oriented` can be `None` (undirected), `"->"`, or `"<-"`.
        """
        def _resolve_node(spec: Union[Node, Tuple[str, Node]]) -> Node:
            if isinstance(spec, (tuple, list)) and len(spec) == 2:
                gid, lname = spec
                if gid in self.gadgets:
                    return self.gadgets[gid].node(lname)
                return f"{gid}.{lname}"
            return str(spec)

        u = _resolve_node(node_a)
        v = _resolve_node(node_b)
        self.add_skeleton_edge(u, v, propagate=True)
        if oriented == "->":
            return self.orient_edge(u, v)
        if oriented == "<-":
            return self.orient_edge(v, u)
        return {"status": True, "message": f"Added undirected link {u} -- {v}."}

    def remove_gadget(self, instance_id: str) -> Dict[str, Any]:
        """
        Completely remove a gadget instance, all of its nodes, internal edges,
        cross-gadget link edges, and associated constraints, then recompute induced orientations.
        """
        if instance_id not in self.gadgets:
            raise ValueError(f"Unknown gadget instance '{instance_id}'.")
        inst = self.gadgets.pop(instance_id)
        removed_nodes = list(inst.node_map.values())
        for n in removed_nodes:
            if n in self.nodes:
                self.remove_node(n, propagate=False)
        self._propagate()
        return {
            "status": True,
            "removed_gadget": instance_id,
            "removed_nodes": removed_nodes,
            "message": f"Removed gadget '{instance_id}' ({len(removed_nodes)} nodes) and updated Meek closure.",
        }

    def reset_constraints(self) -> None:
        """Clear all user orientations, forbidden v-structures, and maximal completions."""
        self.user_oriented.clear()
        self.user_forbidden_v.clear()
        self.completion_oriented.clear()
        self.completion_edge_roles.clear()
        self._propagate()

    def clear_completion(self) -> None:
        """Remove edges oriented by `complete_maximal_v_structures`, restoring the partial graph."""
        self.completion_oriented.clear()
        self.completion_edge_roles.clear()
        self._propagate()

    # ------------------------------------------------------------------
    # Queries on Edges and Incident Edge Pairs (Triples)
    # ------------------------------------------------------------------
    def has_edge(self, u: Node, v: Node) -> bool:
        return normalize_undirected(u, v) in self.skeleton

    def get_all_oriented(self) -> Dict[UndirectedEdge, Edge]:
        out = dict(self.user_oriented)
        out.update(self.induced_oriented)
        out.update(self.completion_oriented)
        return out

    def all_incident_triples(self) -> List[Triple]:
        """Return all triples (u, w, v) of two incident edges (u, w) and (v, w) meeting at w."""
        triples: List[Triple] = []
        for w in sorted(self.nodes):
            nbrs = sorted(self.adj[w])
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    triples.append((nbrs[i], w, nbrs[j]))
        return triples

    def unshielded_triples(self) -> List[Triple]:
        """Return all unshielded triples (u, w, v) where (u, v) is NOT in the skeleton."""
        return [t for t in self.all_incident_triples() if not self.has_edge(t[0], t[2])]

    def shielded_triples(self) -> List[Triple]:
        """Return all shielded triples / triangles (u, w, v) where (u, v) IS in the skeleton."""
        return [t for t in self.all_incident_triples() if self.has_edge(t[0], t[2])]

    # ------------------------------------------------------------------
    # Meek Rules, Cycle Prevention & Global DAG Feasibility Engine
    # ------------------------------------------------------------------
    @staticmethod
    def _find_shortest_directed_path(
        nodes: Set[Node], directed_edges: Set[Edge], start: Node, target: Node
    ) -> Optional[List[Node]]:
        """
        Return the shortest directed path `[start, ..., target]` (by number of edges,
        breaking ties lexicographically) using BFS, or `None` if no path exists.
        """
        if start == target:
            return [start]
        adj_dir: Dict[Node, List[Node]] = {n: [] for n in nodes}
        for u, v in sorted(directed_edges):
            adj_dir[u].append(v)
        visited = {start}
        parent: Dict[Node, Node] = {}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            for nxt in adj_dir[curr]:
                if nxt == target:
                    path = [target, curr]
                    p = curr
                    while p in parent:
                        p = parent[p]
                        path.append(p)
                    path.reverse()
                    return path
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = curr
                    queue.append(nxt)
        return None

    @classmethod
    def _has_directed_path(cls, nodes: Set[Node], directed_edges: Set[Edge], start: Node, target: Node) -> bool:
        return cls._find_shortest_directed_path(nodes, directed_edges, start, target) is not None

    @classmethod
    def _find_shortest_directed_cycle(
        cls, nodes: Set[Node], directed_edges: Set[Edge]
    ) -> Optional[List[Node]]:
        """
        Find and return the shortest directed cycle as `[v0, v1, ..., vk, v0]`,
        or `None` if `directed_edges` is acyclic.
        """
        best_cycle: Optional[List[Node]] = None
        for u, v in sorted(directed_edges):
            path = cls._find_shortest_directed_path(nodes, directed_edges, v, u)
            if path is not None:
                cycle = [u] + path  # [u, v, ..., u]
                if best_cycle is None or len(cycle) < len(best_cycle):
                    best_cycle = cycle
                    if len(best_cycle) == 3:
                        break
        return best_cycle

    @staticmethod
    def _is_acyclic(nodes: Set[Node], directed_edges: Set[Edge]) -> bool:
        in_deg: Dict[Node, int] = {n: 0 for n in nodes}
        adj_dir: Dict[Node, List[Node]] = {n: [] for n in nodes}
        for u, v in directed_edges:
            adj_dir[u].append(v)
            in_deg[v] += 1
        q = [n for n, d in in_deg.items() if d == 0]
        count = 0
        while q:
            curr = q.pop()
            count += 1
            for nxt in adj_dir[curr]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    q.append(nxt)
        return count == len(nodes)

    def _find_state_violation(
        self, state: Dict[UndirectedEdge, Edge], forbidden_v: Set[Triple]
    ) -> Optional[str]:
        """
        If `state` violates acyclicity or a forbidden v-structure constraint, return a specific
        culprit explanation (the shortest directed cycle or the exact violated triple);
        otherwise return `None`.
        """
        dir_set = set(state.values())
        for u, w, v in sorted(forbidden_v):
            if (u, w) in dir_set and (v, w) in dir_set:
                return f"forms forbidden V-structure {u} -> {w} <- {v} at center node {w}"
        if not self._is_acyclic(self.nodes, dir_set):
            cyc = self._find_shortest_directed_cycle(self.nodes, dir_set)
            if cyc:
                return f"creates directed cycle {' -> '.join(cyc)}"
            return "creates a directed cycle"
        return None

    def _meek_closure(
        self,
        oriented: Dict[UndirectedEdge, Edge],
        forbidden_v: Set[Triple],
    ) -> Tuple[bool, Dict[UndirectedEdge, Edge], Dict[UndirectedEdge, str], Dict[UndirectedEdge, str], List[str]]:
        """
        Iterative closure under Meek Rules R1, R2 (generalized directed-path cycle avoidance),
        R3, and R4. Always selects the shortest derivation / path explanation when multiple
        culprits can orient an edge in the same round.
        Returns (is_consistent, closed_oriented, reasons_map, rule_codes_map, log_list).
        """
        curr = dict(oriented)
        reasons: Dict[UndirectedEdge, str] = {}
        codes: Dict[UndirectedEdge, str] = {}
        log: List[str] = []

        violation = self._find_state_violation(curr, forbidden_v)
        if violation is not None:
            log.append(f"Conflict: {violation}")
            return False, curr, reasons, codes, log

        changed = True
        while changed:
            changed = False
            dir_set = set(curr.values())

            # Collect all candidate single-step derivations in the current round and pick the shortest
            # tuple format: (priority_key, edge, directed_pair, code, message)
            candidates: List[Tuple[Tuple[int, int, str], UndirectedEdge, Edge, str, str]] = []

            for e in sorted(self.skeleton):
                if e in curr:
                    continue
                u, v = e

                # --- Rule R1 (Forbidden V-Structure propagation: 1 culprit neighbor x):
                for a, b in [(u, v), (v, u)]:
                    for x in sorted(self.adj[a]):
                        if x == b:
                            continue
                        if (x, a) in dir_set and normalize_triple(x, a, b) in forbidden_v:
                            msg = (
                                f"R1 (Forbidden V ({x}, {a}, {b}) at center {a} with culprit {x} -> {a}): "
                                f"forces {a} -> {b}"
                            )
                            candidates.append(((1, 3, msg), e, (a, b), "R1", msg))

                # --- Rule R2 (Cycle Avoidance along shortest directed path):
                path_uv = self._find_shortest_directed_path(self.nodes, dir_set, u, v)
                if path_uv is not None:
                    path_str = " -> ".join(path_uv)
                    inter = path_uv[1:-1]
                    via_note = f" via {', '.join(inter)}" if inter else ""
                    msg = (
                        f"R2 (Cycle avoidance along directed path {path_str}{via_note}): "
                        f"forces {u} -> {v} (since {v} -> {u} would close cycle {path_str} -> {u})"
                    )
                    candidates.append(((2, len(path_uv), msg), e, (u, v), "R2", msg))

                path_vu = self._find_shortest_directed_path(self.nodes, dir_set, v, u)
                if path_vu is not None:
                    path_str = " -> ".join(path_vu)
                    inter = path_vu[1:-1]
                    via_note = f" via {', '.join(inter)}" if inter else ""
                    msg = (
                        f"R2 (Cycle avoidance along directed path {path_str}{via_note}): "
                        f"forces {v} -> {u} (since {u} -> {v} would close cycle {path_str} -> {v})"
                    )
                    candidates.append(((2, len(path_vu), msg), e, (v, u), "R2", msg))

                # --- Rule R3 (Diamond with unshielded common neighbors b, c):
                for a, d in [(u, v), (v, u)]:
                    common = sorted((self.adj[a] & self.adj[d]) - {a, d})
                    for i in range(len(common)):
                        for j in range(i + 1, len(common)):
                            b, c = common[i], common[j]
                            if self.has_edge(b, c):
                                continue
                            if normalize_undirected(a, b) in curr or normalize_undirected(a, c) in curr:
                                continue
                            if (b, d) in dir_set and (c, d) in dir_set:
                                if normalize_triple(b, a, c) in forbidden_v:
                                    msg = (
                                        f"R3 (Diamond on ({a}, {b}, {c}, {d}) with forbidden V ({b}, {a}, {c}), "
                                        f"undirected {a}-{b}, {a}-{c}, and colliders {b} -> {d} <- {c}): "
                                        f"forces {a} -> {d}"
                                    )
                                    candidates.append(((3, 4, msg), e, (a, d), "R3", msg))

                # --- Rule R4 (Discriminating path / chain d -> c -> b adjacent to a):
                for a, b in [(u, v), (v, u)]:
                    for c in sorted(self.adj[a] & self.adj[b]):
                        if (c, b) not in dir_set:
                            continue
                        for d in sorted(self.adj[a] & self.adj[c]):
                            if d == b or self.has_edge(b, d):
                                continue
                            if (d, c) in dir_set and normalize_triple(d, a, b) in forbidden_v:
                                msg = (
                                    f"R4 (Chain {d} -> {c} -> {b} with {a} adjacent to ({d}, {c}, {b}) "
                                    f"and forbidden V ({d}, {a}, {b})): forces {a} -> {b}"
                                )
                                candidates.append(((4, 4, msg), e, (a, b), "R4", msg))

            if candidates:
                candidates.sort(key=lambda item: item[0])
                _, best_e, best_dir, best_code, best_msg = candidates[0]
                curr[best_e] = best_dir
                reasons[best_e] = best_msg
                codes[best_e] = best_code
                log.append(best_msg)
                changed = True
                violation = self._find_state_violation(curr, forbidden_v)
                if violation is not None:
                    log.append(f"Conflict: {violation}")
                    return False, curr, reasons, codes, log

        return True, curr, reasons, codes, log

    def _explain_hypothetical_contradiction(
        self,
        base_oriented: Dict[UndirectedEdge, Edge],
        hypothetical_edges: Dict[UndirectedEdge, Edge],
        forbidden_v: Set[Triple],
    ) -> str:
        """
        Produce a concise, human-readable explanation of the exact culprit nodes, Meek rules,
        and cycle/forbidden-V contradiction triggered when adding `hypothetical_edges`
        to `base_oriented` under `forbidden_v`.
        """
        trial = dict(base_oriented)
        trial.update(hypothetical_edges)
        ok, closed, _, _, log = self._meek_closure(trial, forbidden_v)
        if not ok:
            steps = [s for s in log if not s.startswith("Conflict: ")]
            conflict = next((s[len("Conflict: "):] for s in reversed(log) if s.startswith("Conflict: ")), None)
            if conflict is None:
                conflict = self._find_state_violation(closed, forbidden_v) or "forces a contradiction"
            if not steps:
                return f"immediately {conflict}"
            step_str = "; then ".join(steps)
            return f"via [{step_str}], which {conflict}"

        # If Meek closure alone did not fail, find a minimal branching witness showing why both directions
        # of some remaining undirected edge fail under Meek closure.
        remaining = [e for e in sorted(self.skeleton) if e not in closed]
        for e2 in remaining:
            x, y = e2
            t1 = dict(closed)
            t1[e2] = (x, y)
            ok1, _, _, _, log1 = self._meek_closure(t1, forbidden_v)
            t2 = dict(closed)
            t2[e2] = (y, x)
            ok2, _, _, _, log2 = self._meek_closure(t2, forbidden_v)
            if not ok1 and not ok2:
                c1 = next((s[len("Conflict: "):] for s in reversed(log1) if s.startswith("Conflict: ")), "contradiction")
                c2 = next((s[len("Conflict: "):] for s in reversed(log2) if s.startswith("Conflict: ")), "contradiction")
                s1 = [s for s in log1 if not s.startswith("Conflict: ")]
                s2 = [s for s in log2 if not s.startswith("Conflict: ")]
                d1 = f"{'; '.join(s1)} -> {c1}" if s1 else c1
                d2 = f"{'; '.join(s2)} -> {c2}" if s2 else c2
                prefix = f"first induces [{'; '.join(log)}], leaving " if log else "leaves "
                return (
                    f"{prefix}edge ({x}, {y}) with no valid orientation "
                    f"({x} -> {y}: {d1}; whereas {y} -> {x}: {d2})"
                )

        if log:
            return f"forces [{'; '.join(log)}], which admits no valid acyclic DAG completion"
        return "admits no valid acyclic DAG completion under current forbidden V-structures"

    def _find_valid_dag(
        self,
        oriented: Dict[UndirectedEdge, Edge],
        forbidden_v: Set[Triple],
        prefer_random: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Optional[Dict[UndirectedEdge, Edge]]:
        """
        Backtracking solver with Meek-closure unit propagation at every search node.
        Returns a complete acyclic orientation mapping every skeleton edge to (u, v),
        or None if no valid DAG completion exists.
        """
        ok, closed, _, _, _ = self._meek_closure(oriented, forbidden_v)
        if not ok:
            return None
        if len(closed) == len(self.skeleton):
            return closed

        remaining = [e for e in sorted(self.skeleton) if e not in closed]
        if prefer_random and rng is not None:
            rng.shuffle(remaining)
            e = remaining[0]
            dirs = [(e[0], e[1]), (e[1], e[0])]
            rng.shuffle(dirs)
        else:
            dir_nodes = {n for edge in closed.values() for n in edge}
            e = max(remaining, key=lambda pair: (pair[0] in dir_nodes) + (pair[1] in dir_nodes))
            dirs = [(e[0], e[1]), (e[1], e[0])]

        for src, dst in dirs:
            trial = dict(closed)
            trial[e] = (src, dst)
            res = self._find_valid_dag(trial, forbidden_v, prefer_random=prefer_random, rng=rng)
            if res is not None:
                return res
        return None

    def _propagate(self) -> None:
        """
        Recompute all induced orientations from `user_oriented` and `user_forbidden_v`:
          1. Direct Meek Rules (R1, R2, R3, R4).
          2. Cycle-Lookahead Propagation: if orienting u -> v triggers a cycle/contradiction
             under Meek rules, v -> u is immediately induced with the shortest culprit trace.
          3. Global DAG Backbone Propagation: if orienting u -> v admits zero valid DAG
             completions, v -> u is induced with a concrete culprit explanation.
        """
        self.induced_oriented.clear()
        self.induced_reasons.clear()
        self._witness_dags.clear()
        self.induced_rule_codes.clear()
        self.propagation_log.clear()

        base = dict(self.user_oriented)
        ok, curr, reasons, codes, log = self._meek_closure(base, self.user_forbidden_v)
        if not ok:
            violation = next((s[len("Conflict: "):] for s in reversed(log) if s.startswith("Conflict: ")), None)
            detail = f": {violation}" if violation else "."
            raise ValueError(f"Current constraints are inconsistent (no valid DAG exists){detail}")

        for e, d in curr.items():
            if e not in base:
                self.induced_oriented[e] = d
                self.induced_reasons[e] = reasons[e]
                self.induced_rule_codes[e] = codes[e]
        self.propagation_log.extend(log)

        changed = True
        while changed:
            changed = False
            # Gather all lookahead candidates and pick the one with the shortest contradiction trace
            lookahead_candidates: List[Tuple[int, int, str, UndirectedEdge, Edge, str]] = []
            for e in sorted(self.skeleton):
                if e in curr:
                    continue
                u, v = e

                # Test hypothetical u -> v under Meek closure
                trial_uv = dict(curr)
                trial_uv[e] = (u, v)
                ok_uv, _, _, _, log_uv = self._meek_closure(trial_uv, self.user_forbidden_v)
                if not ok_uv:
                    trace_explanation = self._explain_hypothetical_contradiction(
                        curr, {e: (u, v)}, self.user_forbidden_v
                    )
                    msg = (
                        f"Lookahead Cycle Avoidance: orienting {u} -> {v} {trace_explanation}, "
                        f"so {v} -> {u} is forced"
                    )
                    lookahead_candidates.append((len(log_uv), len(msg), msg, e, (v, u), msg))
                    continue

                # Test hypothetical v -> u under Meek closure
                trial_vu = dict(curr)
                trial_vu[e] = (v, u)
                ok_vu, _, _, _, log_vu = self._meek_closure(trial_vu, self.user_forbidden_v)
                if not ok_vu:
                    trace_explanation = self._explain_hypothetical_contradiction(
                        curr, {e: (v, u)}, self.user_forbidden_v
                    )
                    msg = (
                        f"Lookahead Cycle Avoidance: orienting {v} -> {u} {trace_explanation}, "
                        f"so {u} -> {v} is forced"
                    )
                    lookahead_candidates.append((len(log_vu), len(msg), msg, e, (u, v), msg))
                    continue

            if lookahead_candidates:
                lookahead_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
                _, _, _, best_e, best_dir, best_msg = lookahead_candidates[0]
                curr[best_e] = best_dir
                self.induced_oriented[best_e] = best_dir
                self.induced_reasons[best_e] = best_msg
                self.induced_rule_codes[best_e] = "Lookahead"
                self.propagation_log.append(best_msg)
                changed = True
                ok, curr, r_new, c_new, l_new = self._meek_closure(curr, self.user_forbidden_v)
                for e2, d2 in curr.items():
                    if e2 not in base and e2 not in self.induced_oriented:
                        self.induced_oriented[e2] = d2
                        self.induced_reasons[e2] = r_new[e2]
                        self.induced_rule_codes[e2] = c_new[e2]
                self.propagation_log.extend(l_new)
                continue

            # Global DAG satisfiability check (accelerated with witness DAG cache)
            for e in sorted(self.skeleton):
                if e in curr:
                    continue
                u, v = e
                trial_uv = dict(curr)
                trial_uv[e] = (u, v)
                if not any((u, v) in wdag for wdag in self._witness_dags):
                    dag_uv = self._find_valid_dag(trial_uv, self.user_forbidden_v)
                    if dag_uv is None:
                        curr[e] = (v, u)
                        trace_explanation = self._explain_hypothetical_contradiction(
                            curr, {e: (u, v)}, self.user_forbidden_v
                        )
                        msg = (
                            f"Global Cycle Avoidance: orienting {u} -> {v} {trace_explanation}, "
                            f"so {v} -> {u} is forced"
                        )
                        self.induced_oriented[e] = (v, u)
                        self.induced_reasons[e] = msg
                        self.induced_rule_codes[e] = "Backbone"
                        self.propagation_log.append(msg)
                        changed = True
                        ok, curr, r_new, c_new, l_new = self._meek_closure(curr, self.user_forbidden_v)
                        for e2, d2 in curr.items():
                            if e2 not in base and e2 not in self.induced_oriented:
                                self.induced_oriented[e2] = d2
                                self.induced_reasons[e2] = r_new[e2]
                                self.induced_rule_codes[e2] = c_new[e2]
                        self.propagation_log.extend(l_new)
                        break
                    else:
                        self._witness_dags.append(set(dag_uv.values()))

                trial_vu = dict(curr)
                trial_vu[e] = (v, u)
                if not any((v, u) in wdag for wdag in self._witness_dags):
                    dag_vu = self._find_valid_dag(trial_vu, self.user_forbidden_v)
                    if dag_vu is None:
                        curr[e] = (u, v)
                        trace_explanation = self._explain_hypothetical_contradiction(
                            curr, {e: (v, u)}, self.user_forbidden_v
                        )
                        msg = (
                            f"Global Cycle Avoidance: orienting {v} -> {u} {trace_explanation}, "
                            f"so {u} -> {v} is forced"
                        )
                        self.induced_oriented[e] = (u, v)
                        self.induced_reasons[e] = msg
                        self.induced_rule_codes[e] = "Backbone"
                        self.propagation_log.append(msg)
                        changed = True
                        ok, curr, r_new, c_new, l_new = self._meek_closure(curr, self.user_forbidden_v)
                        for e2, d2 in curr.items():
                            if e2 not in base and e2 not in self.induced_oriented:
                                self.induced_oriented[e2] = d2
                                self.induced_reasons[e2] = r_new[e2]
                                self.induced_rule_codes[e2] = c_new[e2]
                        self.propagation_log.extend(l_new)
                        break
                    else:
                        self._witness_dags.append(set(dag_vu.values()))

    # ------------------------------------------------------------------
    # User Commands: Orient / Unorient Edge & Forbid / Unforbid V-Structure
    # ------------------------------------------------------------------
    def orient_edge(self, u: Node, v: Node) -> Dict[str, Any]:
        """
        Command to orient edge `u -> v` and propagate Meek rules & cycle-avoidance.
        If `u -> v` would generate a cycle under the current constraints, the opposite
        orientation `v -> u` is already present in `induced_oriented`, and this method
        raises a clear ValueError explaining why `u -> v` is blocked.
        """
        u, v = str(u), str(v)
        e = normalize_undirected(u, v)
        if e not in self.skeleton:
            raise ValueError(f"Edge ({u}, {v}) is not in the skeleton.")

        if self.completion_oriented:
            self.clear_completion()

        all_oriented = self.get_all_oriented()
        if e in all_oriented:
            cur_u, cur_v = all_oriented[e]
            if (cur_u, cur_v) == (u, v):
                if e in self.user_oriented:
                    return {
                        "status": True,
                        "message": f"Edge {u} -> {v} is already oriented by user.",
                        "new_induced": [],
                    }
                return {
                    "status": True,
                    "message": f"Edge {u} -> {v} was already induced ({self.induced_reasons.get(e, 'by constraints')}).",
                    "new_induced": [],
                }
            else:
                if e in self.user_oriented:
                    reason = f"explicitly oriented by user as {cur_u} -> {cur_v}"
                else:
                    reason = self.induced_reasons.get(e, "induced by constraints")
                raise ValueError(
                    f"Cannot orient {u} -> {v}: this edge is already oriented as {cur_u} -> {cur_v} ({reason})."
                )

        prev_induced = set(self.induced_oriented.values())
        self.user_oriented[e] = (u, v)
        try:
            self._propagate()
        except ValueError as exc:
            del self.user_oriented[e]
            self._propagate()
            raise ValueError(f"Cannot orient {u} -> {v}: {exc}") from exc

        newly_induced = [
            f"{src} -> {dst} [{self.induced_reasons[normalize_undirected(src, dst)]}]"
            for src, dst in sorted(set(self.induced_oriented.values()) - prev_induced)
        ]
        return {
            "status": True,
            "message": f"Oriented {u} -> {v} (user). Induced {len(newly_induced)} additional edge(s).",
            "new_induced": newly_induced,
        }

    def unorient_edge(self, u: Node, v: Node) -> Dict[str, Any]:
        """Remove a user-chosen orientation on edge (u, v) and recompute induced edges."""
        e = normalize_undirected(str(u), str(v))
        if e not in self.user_oriented:
            raise ValueError(f"Edge {e} does not have a user-chosen orientation to remove.")
        if self.completion_oriented:
            self.clear_completion()
        removed = self.user_oriented.pop(e)
        self._propagate()
        return {"status": True, "message": f"Removed user orientation {removed[0]} -> {removed[1]}."}

    def _describe_edge_reason(self, e: UndirectedEdge, src: Node, dst: Node) -> str:
        """Return a concise explanation of why edge `e` is oriented as `src -> dst`."""
        if e in self.user_oriented:
            return f"{src} -> {dst} (user-chosen)"
        if e in self.induced_oriented:
            return f"{src} -> {dst} ({self.induced_reasons.get(e, 'induced')})"
        if e in self.completion_oriented:
            role = self.completion_edge_roles.get(e, "COMPLETION")
            return f"{src} -> {dst} ({role})"
        return f"{src} -> {dst}"

    def classify_triple(self, *args: Any) -> Dict[str, Any]:
        """
        Classify a pair of incident edges specified either as `(u, w, v)` (centered at `w`)
        or as two incident edges `((u, w), (v, w))`.
        Clearly distinguishes whether the incident edge pair CAN have a no-v-structure
        constraint (`can_forbid=True`) or CANNOT (`can_forbid=False`), and why (including
        exact culprit nodes, paths, or rules).
        """
        t = resolve_triple_args(*args)
        u, w, v = t
        if not (self.has_edge(u, w) and self.has_edge(v, w)):
            raise ValueError(f"Both ({u}, {w}) and ({v}, {w}) must be edges in the skeleton.")

        # 1. Shielded triple (triangle)
        if self.has_edge(u, v):
            return {
                "triple": t,
                "edges": ((u, w), (v, w)),
                "category": "SHIELDED_TRIANGLE",
                "can_forbid": False,
                "can_form_v": False,
                "explanation": (
                    f"Cannot forbid V-structure on ({u}, {w}, {v}): chord ({u}, {v}) exists in the skeleton "
                    f"(triangle {u}-{w}-{v}-{u} — no unshielded V-structure can form)."
                ),
            }

        all_oriented = self.get_all_oriented()
        dir_set = set(all_oriented.values())
        e_uw = normalize_undirected(u, w)
        e_vw = normalize_undirected(v, w)

        # 2. Explicitly forbidden by user
        if t in self.user_forbidden_v:
            return {
                "triple": t,
                "edges": ((u, w), (v, w)),
                "category": "FORBIDDEN_USER",
                "can_forbid": False,
                "can_form_v": False,
                "explanation": f"V-structure {u} -> {w} <- {v} is already forbidden by user/gadget constraint.",
            }

        # 3. Already oriented as a collider u -> w <- v
        if (u, w) in dir_set and (v, w) in dir_set:
            if e_uw in self.completion_oriented or e_vw in self.completion_oriented:
                cat = "ACTIVE_V_COMPLETION"
                exp = (
                    f"Active V-structure {u} -> {w} <- {v} (formed during maximal V-structure completion: "
                    f"{self._describe_edge_reason(e_uw, u, w)}; {self._describe_edge_reason(e_vw, v, w)})."
                )
            elif e_uw in self.user_oriented and e_vw in self.user_oriented:
                cat = "ACTIVE_V_USER"
                exp = (
                    f"Cannot forbid: both {u} -> {w} and {v} -> {w} are explicitly oriented toward center {w} by user."
                )
            else:
                cat = "ACTIVE_V_INDUCED"
                r_uw = self._describe_edge_reason(e_uw, u, w)
                r_vw = self._describe_edge_reason(e_vw, v, w)
                exp = (
                    f"Cannot forbid V-structure on ({u}, {w}, {v}): collider {u} -> {w} <- {v} is forced because "
                    f"[{r_uw}] and [{r_vw}]."
                )
            return {
                "triple": t,
                "edges": ((u, w), (v, w)),
                "category": cat,
                "can_forbid": False,
                "can_form_v": True,
                "explanation": exp,
            }

        # 4. Already oriented as a non-collider (w -> u or w -> v)
        if (w, u) in dir_set or (w, v) in dir_set:
            out_reasons = []
            if (w, u) in dir_set:
                out_reasons.append(self._describe_edge_reason(e_uw, w, u))
            if (w, v) in dir_set:
                out_reasons.append(self._describe_edge_reason(e_vw, w, v))
            return {
                "triple": t,
                "edges": ((u, w), (v, w)),
                "category": "PREVENTED_NON_COLLIDER",
                "can_forbid": False,
                "can_form_v": False,
                "explanation": (
                    f"Cannot form V-structure (already a non-collider at {w}): "
                    f"{'; '.join(out_reasons)} points away from center {w}."
                ),
            }

        # 5. Check global DAG feasibility of forming V vs avoiding V (using witness DAG cache when available)
        valid_witnesses = [
            wdag for wdag in self._witness_dags
            if dir_set.issubset(wdag)
        ]
        if any((u, w) in wdag and (v, w) in wdag for wdag in valid_witnesses):
            can_form_v = True
        else:
            trial_v = dict(all_oriented)
            trial_v[e_uw] = (u, w)
            trial_v[e_vw] = (v, w)
            dag_v = self._find_valid_dag(trial_v, self.user_forbidden_v)
            can_form_v = dag_v is not None
            if dag_v is not None:
                self._witness_dags.append(set(dag_v.values()))

        if any(not ((u, w) in wdag and (v, w) in wdag) for wdag in valid_witnesses):
            can_avoid_v = True
        else:
            dag_av = self._find_valid_dag(all_oriented, self.user_forbidden_v | {t})
            can_avoid_v = dag_av is not None
            if dag_av is not None:
                self._witness_dags.append(set(dag_av.values()))

        if not can_form_v:
            culprit_trace = self._explain_hypothetical_contradiction(
                all_oriented, {e_uw: (u, w), e_vw: (v, w)}, self.user_forbidden_v
            )
            return {
                "triple": t,
                "edges": ((u, w), (v, w)),
                "category": "INDUCED_FORBIDDEN_V",
                "can_forbid": False,
                "can_form_v": False,
                "explanation": (
                    f"Cannot form V-structure (Induced No-V-Structure): orienting {u} -> {w} <- {v} "
                    f"{culprit_trace}."
                ),
            }

        if not can_avoid_v:
            culprit_trace = self._explain_hypothetical_contradiction(
                all_oriented, {}, self.user_forbidden_v | {t}
            )
            return {
                "triple": t,
                "edges": ((u, w), (v, w)),
                "category": "ACTIVE_V_INDUCED",
                "can_forbid": False,
                "can_form_v": True,
                "explanation": (
                    f"Cannot forbid V-structure on ({u}, {w}, {v}): forbidding {u} -> {w} <- {v} "
                    f"{culprit_trace}."
                ),
            }

        return {
            "triple": t,
            "edges": ((u, w), (v, w)),
            "category": "ALLOWED_OPEN",
            "can_forbid": True,
            "can_form_v": True,
            "explanation": (
                f"Allowed / Open unshielded pair ({u}-{w}, {v}-{w}): can forbid V-structure "
                f"OR can form V-structure {u} -> {w} <- {v}."
            ),
        }

    def forbid_v_structure(self, *args: Any) -> Dict[str, Any]:
        """
        Forbid a v-structure on an incident edge pair, passed either as three nodes `(u, w, v)`
        (centered at `w`) or as two incident edges `((u, w), (v, w))`.
        Raises a descriptive ValueError if the pair cannot have a no-v-structure constraint.
        """
        if self.completion_oriented:
            self.clear_completion()

        info = self.classify_triple(*args)
        t = info["triple"]
        if not info["can_forbid"]:
            raise ValueError(info["explanation"])

        prev_induced = set(self.induced_oriented.values())
        self.user_forbidden_v.add(t)
        try:
            self._propagate()
        except ValueError as exc:
            self.user_forbidden_v.remove(t)
            self._propagate()
            raise ValueError(f"Cannot forbid V-structure on {t}: {exc}") from exc

        newly_induced = [
            f"{src} -> {dst} [{self.induced_reasons[normalize_undirected(src, dst)]}]"
            for src, dst in sorted(set(self.induced_oriented.values()) - prev_induced)
        ]
        return {
            "status": True,
            "triple": t,
            "message": f"Forbade V-structure on ({t[0]}, {t[1]}, {t[2]}). Induced {len(newly_induced)} edge(s).",
            "new_induced": newly_induced,
        }

    def unforbid_v_structure(self, *args: Any) -> Dict[str, Any]:
        """Remove a user-specified no-v-structure constraint and recompute induced orientations."""
        t = resolve_triple_args(*args)
        if t not in self.user_forbidden_v:
            raise ValueError(f"Triple {t} is not in user_forbidden_v.")
        if self.completion_oriented:
            self.clear_completion()
        self.user_forbidden_v.remove(t)
        self._propagate()
        return {"status": True, "triple": t, "message": f"Removed no-V-structure constraint on {t}."}

    # ------------------------------------------------------------------
    # Inclusion-Maximal V-Structure DAG Completion
    # ------------------------------------------------------------------
    def complete_maximal_v_structures(
        self, seed: Optional[int] = None, apply_to_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Completely direct the partially directed graph into a valid DAG D* such that
        its final set of v-structures V(D*) is inclusion-maximal among all valid DAG
        completions of the current state (user_oriented + induced_oriented + user_forbidden_v).
        """
        self.clear_completion()
        rng = random.Random(seed)

        base_oriented = dict(self.user_oriented)
        base_oriented.update(self.induced_oriented)
        base_dir_set = set(base_oriented.values())

        initial_v_structures = {
            t for t in self.unshielded_triples()
            if (t[0], t[1]) in base_dir_set and (t[2], t[1]) in base_dir_set
        }

        candidates = [
            t for t in self.unshielded_triples()
            if t not in self.user_forbidden_v and t not in initial_v_structures
        ]
        rng.shuffle(candidates)

        curr_oriented = dict(base_oriented)
        v_edges_added: Set[UndirectedEdge] = set()
        added_v_structures: List[Triple] = []

        for u, w, v in candidates:
            e_uw = normalize_undirected(u, w)
            e_vw = normalize_undirected(v, w)
            if curr_oriented.get(e_uw) == (w, u) or curr_oriented.get(e_vw) == (w, v):
                continue
            trial = dict(curr_oriented)
            trial[e_uw] = (u, w)
            trial[e_vw] = (v, w)
            ok, closed, _, _, _ = self._meek_closure(trial, self.user_forbidden_v)
            if ok and self._find_valid_dag(closed, self.user_forbidden_v, prefer_random=True, rng=rng) is not None:
                if e_uw not in base_oriented:
                    v_edges_added.add(e_uw)
                if e_vw not in base_oriented:
                    v_edges_added.add(e_vw)
                curr_oriented = closed
                added_v_structures.append((u, w, v))

        full_dag = self._find_valid_dag(curr_oriented, self.user_forbidden_v, prefer_random=True, rng=rng)
        if full_dag is None:
            raise RuntimeError("Failed to find a valid DAG completion.")

        final_dir_set = set(full_dag.values())
        final_v_structures = {
            t for t in self.unshielded_triples()
            if (t[0], t[1]) in final_dir_set and (t[2], t[1]) in final_dir_set
        }

        required_for_final_v = dict(base_oriented)
        for a, m, b in final_v_structures:
            required_for_final_v[normalize_undirected(a, m)] = (a, m)
            required_for_final_v[normalize_undirected(b, m)] = (b, m)

        for t in self.unshielded_triples():
            if t in final_v_structures:
                continue
            u, w, v = t
            e_uw = normalize_undirected(u, w)
            e_vw = normalize_undirected(v, w)
            if required_for_final_v.get(e_uw) == (w, u) or required_for_final_v.get(e_vw) == (w, v):
                continue
            trial = dict(required_for_final_v)
            trial[e_uw] = (u, w)
            trial[e_vw] = (v, w)
            if self._find_valid_dag(trial, self.user_forbidden_v) is not None:
                raise AssertionError(f"Inclusion-maximality violated: {t} could still be added!")

        if apply_to_graph:
            for e, d in full_dag.items():
                if e not in base_oriented:
                    self.completion_oriented[e] = d
                    self.completion_edge_roles[e] = (
                        "COMPLETION_V" if e in v_edges_added else "COMPLETION_INDUCED"
                    )

        return {
            "initial_v_structures": sorted(initial_v_structures),
            "added_v_structures": sorted(final_v_structures - initial_v_structures),
            "final_v_structures": sorted(final_v_structures),
            "dag_edges": sorted(full_dag.values()),
            "is_inclusion_maximal_verified": True,
        }

    # ------------------------------------------------------------------
    # Summary Tables & Layout Computation
    # ------------------------------------------------------------------
    def get_triples_summary(self, include_non_collider: bool = False) -> List[Dict[str, Any]]:
        """
        Return classification dicts for incident edge pairs.
        By default (`include_non_collider=False`), excludes triples of status
        `"PREVENTED_NON_COLLIDER"` ("Cannot Forbid (Already Non-Collider)").
        """
        out = [self.classify_triple(t) for t in self.all_incident_triples()]
        if not include_non_collider:
            out = [info for info in out if info["category"] != "PREVENTED_NON_COLLIDER"]
        return out

    def _compute_node_positions(self, radius: float = 220.0) -> Dict[Node, Tuple[float, float]]:
        """
        Return 2D positions for all graph nodes:
          - Nodes with `custom_positions` (such as nodes belonging to a `GadgetInstance`)
            use their exact canonical gadget geometry.
          - Any remaining free nodes are arranged in a circle.
        """
        pos: Dict[Node, Tuple[float, float]] = dict(self.custom_positions)
        remaining = [n for n in sorted(self.nodes) if n not in pos]
        m = len(remaining)
        if m == 1 and not pos:
            pos[remaining[0]] = (0.0, 0.0)
        elif m > 0:
            offset_x = 380.0 * len(self.gadgets) if self.gadgets else 0.0
            for idx, node in enumerate(remaining):
                angle = -math.pi / 2 + (2 * math.pi * idx) / m
                pos[node] = (offset_x + radius * math.cos(angle), radius * math.sin(angle))
        return pos

    # ------------------------------------------------------------------
    # PyVis Network Construction & Rigid-Block Gadget Dragging
    # ------------------------------------------------------------------
    def build_pyvis_network(
        self,
        height: str = "560px",
        width: str = "100%",
        show_triple_badges: bool = True,
        show_shielded_badges: bool = True,
        physics: bool = False,
        visible_triple_categories: Optional[Set[str]] = None,
    ) -> Any:
        """
        Build and return a `pyvis.network.Network` instance representing the current graph.
        Includes `gadget_id` metadata on every gadget node and corner badge so the interactive
        canvas can drag an entire gadget as a rigid block!
        By default (`visible_triple_categories=None`), triple annotation nodes and edges start
        hidden (`hidden=True`) so they are synchronized with the unchecked annotation checkboxes.
        """
        NetCls = PyVisNetwork if _HAS_PYVIS else _FallbackPyVisNetwork
        net = NetCls(
            height=height,
            width=width,
            directed=True,
            notebook=True,
            bgcolor="#ffffff",
            font_color="#0f172a",
            cdn_resources="remote",
        )

        pos = self._compute_node_positions(radius=230.0)
        all_triple_infos = self.get_triples_summary(include_non_collider=True)

        active_colliders = {
            info["triple"][1]
            for info in all_triple_infos
            if info["category"] in ("ACTIVE_V_USER", "ACTIVE_V_INDUCED", "ACTIVE_V_COMPLETION")
        }

        # 1. Add main variable nodes
        for n in sorted(self.nodes):
            x, y = pos[n]
            gad_id = self.node_gadget.get(n)
            centered_here = [
                info for info in all_triple_infos
                if info["triple"][1] == n and info["category"] != "PREVENTED_NON_COLLIDER"
            ]
            tooltip_lines = [f"<b>Node {html.escape(n)}</b>"]
            if gad_id:
                inst = self.gadgets[gad_id]
                tooltip_lines.append(f"Gadget: <b>{html.escape(gad_id)}</b> ({html.escape(inst.template.name)})")
            if centered_here:
                tooltip_lines.append("<hr style='margin:4px 0;'/><b>Incident Edge Pairs at " + html.escape(n) + ":</b>")
                for info in centered_here:
                    u, w, v = info["triple"]
                    st = TRIPLE_STYLES[info["category"]]["label"]
                    tooltip_lines.append(
                        f"&bull; ({u}, {w}, {v}): <b>{st}</b><br/>&nbsp;&nbsp;<i>{html.escape(info['explanation'])}</i>"
                    )

            is_collider = n in active_colliders
            if gad_id and gad_id in self.gadgets:
                pal = GADGET_PALETTE[self.gadgets[gad_id].palette_idx]
                default_bg, default_border = pal["bg"], pal["border"]
            else:
                default_bg, default_border = "#eff6ff", "#1d4ed8"

            net.add_node(
                n,
                label=n,
                title="<br/>".join(tooltip_lines),
                x=x,
                y=y,
                gadget_id=gad_id or "",
                shape="circle",
                size=30,
                physics=physics,
                color={
                    "background": "#dcfce7" if is_collider else default_bg,
                    "border": "#16a34a" if is_collider else default_border,
                    "highlight": {
                        "background": "#bbf7d0" if is_collider else "#dbeafe",
                        "border": "#15803d" if is_collider else default_border,
                    },
                },
                borderWidth=3.5 if is_collider else 2.5,
                font={"size": 16, "color": "#0f172a", "face": "Inter, system-ui, sans-serif", "bold": True},
            )

        # 2. Add skeleton / directed edges
        for e in sorted(self.skeleton):
            u, v = e
            if e in self.user_oriented:
                src, dst = self.user_oriented[e]
                style = EDGE_STYLES["USER"]
                edge_label = "user"
                tooltip = f"<b>User-Chosen Orientation:</b> {src} &rarr; {dst}"
            elif e in self.induced_oriented:
                src, dst = self.induced_oriented[e]
                style = EDGE_STYLES["INDUCED"]
                code = self.induced_rule_codes.get(e, "Meek")
                edge_label = f"induced ({code})"
                reason = html.escape(self.induced_reasons.get(e, "Induced by Meek rules / cycle avoidance"))
                tooltip = f"<b>Induced Orientation ({code}):</b> {src} &rarr; {dst}<br/><i>{reason}</i>"
            elif e in self.completion_oriented:
                src, dst = self.completion_oriented[e]
                role = self.completion_edge_roles.get(e, "COMPLETION_INDUCED")
                style = EDGE_STYLES[role]
                edge_label = "max-V" if role == "COMPLETION_V" else "DAG-fill"
                tooltip = f"<b>{style['label_tag']}:</b> {src} &rarr; {dst}"
            else:
                src, dst = u, v
                style = EDGE_STYLES["UNDIRECTED"]
                edge_label = ""
                tooltip = f"<b>Undirected Edge:</b> {u} &mdash; {v} (freely directable)"

            edge_kwargs: Dict[str, Any] = {
                "title": tooltip,
                "color": style["color"],
                "width": style["width"],
                "dashes": style["dashes"],
                "arrows": style["arrows"],
                "smooth": {"enabled": True, "type": "continuous"},
            }
            if edge_label:
                edge_kwargs["label"] = edge_label
                edge_kwargs["font"] = {
                    "size": 11,
                    "color": style["color"],
                    "background": "#ffffff",
                    "strokeWidth": 2,
                    "strokeColor": "#ffffff",
                }
            net.add_edge(src, dst, **edge_kwargs)

        # 3. Add visual corner wedge badges for incident edge pairs (triples)
        if show_triple_badges:
            by_center: Dict[Node, List[Dict[str, Any]]] = {}
            for info in all_triple_infos:
                if info["category"] == "SHIELDED_TRIANGLE" and not show_shielded_badges:
                    continue
                by_center.setdefault(info["triple"][1], []).append(info)

            for w, infos in by_center.items():
                xw, yw = pos[w]
                w_gad = self.node_gadget.get(w, "")
                for idx, info in enumerate(infos):
                    u, _, v = info["triple"]
                    xu, yu = pos[u]
                    xv, yv = pos[v]
                    dux, duy = xu - xw, yu - yw
                    dvx, dvy = xv - xw, yv - yw
                    lu = math.hypot(dux, duy) or 1.0
                    lv = math.hypot(dvx, dvy) or 1.0
                    dux, duy = dux / lu, duy / lu
                    dvx, dvy = dvx / lv, dvy / lv

                    bx, by = dux + dvx, duy + dvy
                    lb = math.hypot(bx, by)
                    if lb < 1e-4:
                        bx, by = -duy, dux
                    else:
                        bx, by = bx / lb, by / lb

                    dist = 56.0 + 14.0 * (idx % 2)
                    tx, ty = xw + dist * bx, yw + dist * by

                    cat = info["category"]
                    tstyle = TRIPLE_STYLES[cat]
                    badge_id = f"triple__{u}_{w}_{v}"

                    short_tag = {
                        "ALLOWED_OPEN": f"Allowed V\n({u},{w},{v})",
                        "FORBIDDEN_USER": f"Forbidden V\n({u},{w},{v})",
                        "INDUCED_FORBIDDEN_V": f"Induced No-V\n({u},{w},{v})",
                        "ACTIVE_V_USER": f"V (User)\n{u}->{w}<-{v}",
                        "ACTIVE_V_INDUCED": f"Forced V (No-Forbid)\n{u}->{w}<-{v}",
                        "ACTIVE_V_COMPLETION": f"V (Maximal)\n{u}->{w}<-{v}",
                        "PREVENTED_NON_COLLIDER": f"Non-Collider\n({u},{w},{v})",
                        "SHIELDED_TRIANGLE": f"Triangle (No-V)\n({u},{w},{v})",
                    }[cat]

                    # If all 3 nodes of the triple belong to the same gadget, attach badge to that gadget block
                    same_gad = (
                        w_gad
                        if (w_gad and self.node_gadget.get(u) == w_gad and self.node_gadget.get(v) == w_gad)
                        else ""
                    )

                    if visible_triple_categories is None:
                        hide_by_default = cat in DEFAULT_HIDDEN_TRIPLE_CATEGORIES
                    else:
                        hide_by_default = cat not in visible_triple_categories

                    net.add_node(
                        badge_id,
                        label=short_tag,
                        title=(
                            f"<b>Incident Pair ({u}-{w}, {v}-{w}) at {w}</b><br/>"
                            f"Status: <b>{tstyle['label']}</b><br/>"
                            f"Can add no-V-structure constraint: <b>{info['can_forbid']}</b><br/>"
                            f"Can form V-structure: <b>{info['can_form_v']}</b><br/>"
                            f"<i>{html.escape(info['explanation'])}</i>"
                        ),
                        x=tx,
                        y=ty,
                        gadget_id=same_gad,
                        triple_category=cat,
                        hidden=hide_by_default,
                        shape=tstyle["shape"],
                        size=11,
                        physics=physics,
                        color={
                            "background": tstyle["bg"],
                            "border": tstyle["color"],
                            "highlight": {"background": tstyle["bg"], "border": tstyle["color"]},
                        },
                        borderWidth=2,
                        font={"size": 9, "color": "#1e293b", "face": "Inter, system-ui, sans-serif"},
                    )
                    for endpoint, weight in [(w, 2.0), (u, 1.0), (v, 1.0)]:
                        net.add_edge(
                            badge_id,
                            endpoint,
                            triple_category=cat,
                            hidden=hide_by_default,
                            color={"color": tstyle["color"], "opacity": 0.35},
                            width=weight,
                            dashes=[2, 4],
                            arrows="",
                            smooth=False,
                        )

        options = {
            "interaction": {"hover": True, "navigationButtons": True, "multiselect": True},
            "physics": {"enabled": physics},
            "edges": {"smooth": {"enabled": True, "type": "continuous"}},
        }
        net.set_options(json.dumps(options))
        return net

    def render_html(
        self,
        title: str = "Bayesian Network State",
        height: str = "540px",
        show_triple_badges: bool = True,
        show_shielded_badges: bool = True,
        visible_triple_categories: Optional[Set[str]] = None,
    ) -> str:
        """
        Generate a complete self-contained HTML view combining:
          1. Visual Legend + Rigid-Block Gadget Dragging Toggle + Synchronized Triple Annotation Toggles
          2. Interactive `pyvis.network` graph canvas (with JS hook to drag gadgets as rigid blocks)
          3. User-Chosen Edge Orientations table
          4. Induced by Meek / Cycle Edge Orientations table
          5. Sortable & Filterable Status Table for Incident Edge Pairs (excluding Already Non-Collider)
        """
        if visible_triple_categories is None:
            visible_cats: Set[str] = set()
        else:
            visible_cats = set(visible_triple_categories)

        net = self.build_pyvis_network(
            height=height,
            show_triple_badges=show_triple_badges,
            show_shielded_badges=show_shielded_badges,
            visible_triple_categories=visible_cats,
        )
        raw_html = net.generate_html()
        toggle_id = f"gadget_drag_toggle_{random.randint(10000, 99999)}"

        # Inject Rigid-Block Gadget Dragging, Synchronized Annotation Toggles, Table Filtering & Sorting JS
        rigid_drag_js = f"""
<script type="text/javascript">
(function() {{
  function attachGadgetRigidDrag() {{
    var netInstance = (typeof network !== "undefined") ? network : null;
    var nodesDS = (typeof nodes !== "undefined") ? nodes : null;
    if (!netInstance) {{
      for (var k in window) {{
        if (k.indexOf("__bn_vis_") === 0 && window[k] && window[k].network) {{
          netInstance = window[k].network;
          nodesDS = window[k].nodes;
        }}
      }}
    }}
    if (!netInstance || !nodesDS || netInstance.__gadgetDragAttached) return;
    netInstance.__gadgetDragAttached = true;

    var dragState = null;
    netInstance.on("dragStart", function(params) {{
      var toggleEl = document.getElementById("{toggle_id}");
      var blockMode = toggleEl ? toggleEl.checked : false;
      if (!blockMode || !params.nodes || params.nodes.length !== 1) {{
        dragState = null;
        return;
      }}
      var draggedId = params.nodes[0];
      var nodeObj = nodesDS.get(draggedId);
      if (!nodeObj || !nodeObj.gadget_id) {{
        dragState = null;
        return;
      }}
      var gid = nodeObj.gadget_id;
      var allItems = nodesDS.get({{
        filter: function(item) {{ return item.gadget_id === gid; }}
      }});
      var memberIds = allItems.map(function(it) {{ return it.id; }});
      var startPositions = netInstance.getPositions(memberIds);
      dragState = {{
        draggedId: draggedId,
        memberIds: memberIds,
        startPositions: startPositions
      }};
    }});

    netInstance.on("dragging", function(params) {{
      if (!dragState) return;
      var curPos = netInstance.getPositions([dragState.draggedId])[dragState.draggedId];
      var origDragged = dragState.startPositions[dragState.draggedId];
      if (!curPos || !origDragged) return;
      var dx = curPos.x - origDragged.x;
      var dy = curPos.y - origDragged.y;
      for (var i = 0; i < dragState.memberIds.length; i++) {{
        var mid = dragState.memberIds[i];
        if (mid === dragState.draggedId) continue;
        var p0 = dragState.startPositions[mid];
        if (p0) {{
          netInstance.moveNode(mid, p0.x + dx, p0.y + dy);
        }}
      }}
    }});

    netInstance.on("dragEnd", function() {{
      dragState = null;
    }});

    var edgesDS = (typeof edges !== "undefined") ? edges : null;
    if (!edgesDS) {{
      for (var k2 in window) {{
        if (k2.indexOf("__bn_vis_") === 0 && window[k2] && window[k2].edges) {{
          edgesDS = window[k2].edges;
        }}
      }}
    }}

    var rootEl = document.getElementById("bn_wrap_{toggle_id}");
    var toggles = rootEl ? rootEl.querySelectorAll(".bn-triple-toggle") : document.querySelectorAll(".bn-triple-toggle");
    var syncSummaryCb = rootEl ? rootEl.querySelector(".bn-sync-summary-toggle") : null;
    var summaryTable = rootEl ? rootEl.querySelector(".bn-triples-table") : null;
    var countBadge = rootEl ? rootEl.querySelector(".bn-triples-count") : null;

    function syncGraphCategory(cats, hide) {{
      var matchedNodes = nodesDS.get({{
        filter: function(n) {{ return n.triple_category && cats.indexOf(n.triple_category) !== -1; }}
      }});
      if (matchedNodes.length > 0) {{
        nodesDS.update(matchedNodes.map(function(n) {{ return {{ id: n.id, hidden: hide }}; }}));
      }}
      if (edgesDS) {{
        var matchedEdges = edgesDS.get({{
          filter: function(e) {{ return e.triple_category && cats.indexOf(e.triple_category) !== -1; }}
        }});
        if (matchedEdges.length > 0) {{
          edgesDS.update(matchedEdges.map(function(e) {{ return {{ id: e.id, hidden: hide }}; }}));
        }}
      }}
    }}

    function updateSummaryTableVisibility() {{
      if (!summaryTable) return;
      var filterActive = syncSummaryCb ? syncSummaryCb.checked : false;
      var activeCats = {{}};
      toggles.forEach(function(cb) {{
        var cats = cb.getAttribute("data-cats").split(",");
        cats.forEach(function(c) {{
          activeCats[c] = cb.checked;
        }});
      }});
      var rows = summaryTable.querySelectorAll("tbody tr[data-cat]");
      var visibleCount = 0;
      rows.forEach(function(row) {{
        var cat = row.getAttribute("data-cat");
        var showRow = !filterActive || !!activeCats[cat];
        row.style.display = showRow ? "" : "none";
        if (showRow) visibleCount += 1;
      }});
      var emptyNotice = summaryTable.querySelector("tbody tr.bn-empty-filter-row");
      if (emptyNotice) {{
        emptyNotice.style.display = (rows.length > 0 && visibleCount === 0) ? "" : "none";
      }}
      if (countBadge) {{
        if (filterActive) {{
          countBadge.textContent = visibleCount + " / " + rows.length + " triples shown";
        }} else {{
          countBadge.textContent = rows.length + " triples";
        }}
      }}
    }}

    // Ensure initial graph visibility matches checkbox states on load
    toggles.forEach(function(cb) {{
      var cats = cb.getAttribute("data-cats").split(",");
      syncGraphCategory(cats, !cb.checked);
      cb.addEventListener("change", function() {{
        syncGraphCategory(cats, !cb.checked);
        updateSummaryTableVisibility();
      }});
    }});

    if (syncSummaryCb) {{
      syncSummaryCb.addEventListener("change", updateSummaryTableVisibility);
    }}
    updateSummaryTableVisibility();

    // Sortable columns for Incident Edge Pairs & V-Structure Constraints Status table
    if (summaryTable) {{
      var headers = summaryTable.querySelectorAll("thead th[data-sort-col]");
      headers.forEach(function(th) {{
        th.addEventListener("click", function() {{
          var colIdx = parseInt(th.getAttribute("data-sort-col"), 10);
          var curDir = th.getAttribute("data-sort-dir") || "none";
          var nextDir = (curDir === "asc") ? "desc" : "asc";
          headers.forEach(function(h) {{
            h.setAttribute("data-sort-dir", "none");
            var ind = h.querySelector(".bn-sort-indicator");
            if (ind) ind.textContent = "⇅";
          }});
          th.setAttribute("data-sort-dir", nextDir);
          var activeInd = th.querySelector(".bn-sort-indicator");
          if (activeInd) activeInd.textContent = (nextDir === "asc") ? "▲" : "▼";

          var tbody = summaryTable.querySelector("tbody");
          var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr[data-cat]"));
          rows.sort(function(ra, rb) {{
            var ca = ra.children[colIdx];
            var cb = rb.children[colIdx];
            var va = (ca ? (ca.getAttribute("data-sort-val") || ca.textContent) : "").trim().toLowerCase();
            var vb = (cb ? (cb.getAttribute("data-sort-val") || cb.textContent) : "").trim().toLowerCase();
            if (va < vb) return (nextDir === "asc") ? -1 : 1;
            if (va > vb) return (nextDir === "asc") ? 1 : -1;
            return 0;
          }});
          rows.forEach(function(r) {{ tbody.appendChild(r); }});
        }});
      }});
    }}
  }}
  setTimeout(attachGadgetRigidDrag, 60);
}})();
</script>
"""
        raw_html = raw_html + rigid_drag_js

        triple_rows = []
        for info in self.get_triples_summary(include_non_collider=False):
            u, w, v = info["triple"]
            cat = info["category"]
            st = TRIPLE_STYLES[cat]
            can_f_sort = "1_yes" if info["can_forbid"] else "2_no"
            can_f_badge = (
                "<span style='background:#dcfce7;color:#166534;padding:2px 6px;border-radius:4px;font-weight:600;'>Yes (Can Forbid)</span>"
                if info["can_forbid"]
                else "<span style='background:#f1f5f9;color:#475569;padding:2px 6px;border-radius:4px;'>No</span>"
            )
            triple_rows.append(
                f"<tr data-cat='{cat}'>"
                f"<td data-sort-val='{html.escape(f'{u},{w},{v}')}' style='padding:6px 8px;border-bottom:1px solid #e2e8f0;font-family:monospace;font-weight:600;'>({html.escape(u)}, {html.escape(w)}, {html.escape(v)})</td>"
                f"<td data-sort-val='{html.escape(f'{u}-{w},{v}-{w}')}' style='padding:6px 8px;border-bottom:1px solid #e2e8f0;font-family:monospace;'>({html.escape(u)}&ndash;{html.escape(w)}), ({html.escape(v)}&ndash;{html.escape(w)})</td>"
                f"<td data-sort-val='{html.escape(st['label'])}' style='padding:6px 8px;border-bottom:1px solid #e2e8f0;'>"
                f"<span style='background:{st['bg']};color:{st['color']};border:1px solid {st['color']};padding:2px 6px;border-radius:4px;font-size:12px;font-weight:600;'>{st['label']}</span>"
                f"</td>"
                f"<td data-sort-val='{can_f_sort}' style='padding:6px 8px;border-bottom:1px solid #e2e8f0;'>{can_f_badge}</td>"
                f"<td data-sort-val='{html.escape(info['explanation'])}' style='padding:6px 8px;border-bottom:1px solid #e2e8f0;font-size:12px;color:#334155;'>{html.escape(info['explanation'])}</td>"
                f"</tr>"
            )

        triple_rows.append(
            "<tr class='bn-empty-filter-row' style='display:none;'>"
            "<td colspan='5' style='padding:8px;color:#64748b;font-style:italic;'>"
            "No triples match the currently checked V-Structure / Triple toggles above.</td></tr>"
        )

        def _scope_for_edge(src: Node, dst: Node) -> str:
            gad_src = self.node_gadget.get(src)
            gad_dst = self.node_gadget.get(dst)
            if gad_src and gad_src == gad_dst:
                return f"Gadget {html.escape(gad_src)}"
            elif gad_src or gad_dst:
                return f"{html.escape(gad_src or 'Free')} &rarr; {html.escape(gad_dst or 'Free')}"
            return "Skeleton"

        user_choice_rows = []
        for idx, (e, (src, dst)) in enumerate(sorted(self.user_oriented.items()), start=1):
            scope_label = _scope_for_edge(src, dst)
            user_choice_rows.append(
                f"<tr>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #e2e8f0;color:#64748b;'>{idx}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #e2e8f0;font-family:monospace;font-weight:600;color:#2563eb;'>"
                f"{html.escape(src)} &rarr; {html.escape(dst)}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #e2e8f0;font-family:monospace;'>"
                f"({html.escape(e[0])}&ndash;{html.escape(e[1])})</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #e2e8f0;font-size:12px;color:#334155;'>{scope_label}</td>"
                f"</tr>"
            )

        if not user_choice_rows:
            user_choice_rows.append(
                "<tr><td colspan='4' style='padding:8px;color:#64748b;font-style:italic;'>"
                "No user-chosen edge orientations yet.</td></tr>"
            )

        induced_rows = []
        for idx, (e, (src, dst)) in enumerate(sorted(self.induced_oriented.items()), start=1):
            scope_label = _scope_for_edge(src, dst)
            rule_code = self.induced_rule_codes.get(e, "Meek")
            reason_str = self.induced_reasons.get(e, "Induced by Meek rules / cycle avoidance")
            induced_rows.append(
                f"<tr>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #e2e8f0;color:#64748b;'>{idx}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #e2e8f0;font-family:monospace;font-weight:600;color:#d97706;'>"
                f"{html.escape(src)} &rarr; {html.escape(dst)}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #e2e8f0;font-family:monospace;'>"
                f"({html.escape(e[0])}&ndash;{html.escape(e[1])})</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #e2e8f0;'>"
                f"<span style='background:#fef3c7;color:#b45309;border:1px solid #f59e0b;padding:2px 6px;border-radius:4px;font-size:11.5px;font-weight:600;'>"
                f"{html.escape(rule_code)}</span></td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #e2e8f0;font-size:12px;color:#334155;'>{scope_label}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #e2e8f0;font-size:12px;color:#334155;'>{html.escape(reason_str)}</td>"
                f"</tr>"
            )

        if not induced_rows:
            induced_rows.append(
                "<tr><td colspan='6' style='padding:8px;color:#64748b;font-style:italic;'>"
                "No orientations induced by Meek rules or cycle avoidance yet.</td></tr>"
            )

        gadget_bar = ""
        if self.gadgets:
            g_pills = []
            for gid, inst in self.gadgets.items():
                pal = GADGET_PALETTE[inst.palette_idx]
                g_pills.append(
                    f"<span style='background:{pal['bg']};color:{pal['border']};border:1.5px solid {pal['border']};"
                    f"padding:2px 8px;border-radius:999px;font-weight:600;font-size:12px;'>"
                    f"{html.escape(gid)} ({html.escape(inst.template.name)})</span>"
                )
            gadget_bar = (
                f"<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;"
                f"font-size:12.5px;margin-bottom:10px;padding:8px 10px;background:#ffffff;border:1px solid #cbd5e1;border-radius:6px;'>"
                f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'><b>Active Gadgets:</b> {' '.join(g_pills)}</div>"
                f"<label style='cursor:pointer;font-weight:600;color:#1e293b;display:flex;align-items:center;gap:6px;'>"
                f"<input type='checkbox' id='{toggle_id}' checked style='cursor:pointer;'/> "
                f"Drag Gadgets as Rigid Block (uncheck to move single nodes)</label>"
                f"</div>"
            )

        def _checked_attr(cat: str) -> str:
            return " checked" if cat in visible_cats else ""

        num_summary_triples = len(triple_rows) - 1  # exclude the hidden empty-filter notice row

        legend_and_tables = f"""
<div id="bn_wrap_{toggle_id}" style="font-family:Inter,system-ui,sans-serif;color:#0f172a;margin:8px 0 16px 0;border:1px solid #cbd5e1;border-radius:10px;padding:14px;background:#f8fafc;">
  <div style="font-size:16px;font-weight:700;margin-bottom:8px;">{html.escape(title)}</div>
  {gadget_bar}
  <div style="display:flex;flex-wrap:wrap;gap:14px;font-size:12px;margin-bottom:10px;padding:8px 10px;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;">
    <div><b>Edges:</b></div>
    <div><span style="display:inline-block;width:22px;height:4px;background:#2563eb;vertical-align:middle;"></span> <b>User-Oriented (&rarr;)</b></div>
    <div><span style="display:inline-block;width:22px;height:0;border-top:3px dashed #d97706;vertical-align:middle;"></span> <b>Induced by Meek / Cycle (&rarr;)</b></div>
    <div><span style="display:inline-block;width:22px;height:0;border-top:3px dotted #7c3aed;vertical-align:middle;"></span> <b>Maximal Completion (&rarr;)</b></div>
    <div><span style="display:inline-block;width:22px;height:3px;background:#64748b;vertical-align:middle;"></span> <b>Undirected (&mdash;)</b></div>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:10px;font-size:12px;margin-bottom:10px;padding:8px 10px;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;align-items:center;">
    <div><b>Toggle V-Structures / Triples:</b></div>
    <label style="cursor:pointer;background:#e0f2fe;color:#0284c7;border:1px solid #0284c7;padding:2px 6px;border-radius:4px;">
      <input type="checkbox" class="bn-triple-toggle"{_checked_attr("ALLOWED_OPEN")} data-cats="ALLOWED_OPEN"/> Allowed V (Can Forbid)
    </label>
    <label style="cursor:pointer;background:#fee2e2;color:#dc2626;border:1px solid #dc2626;padding:2px 6px;border-radius:4px;">
      <input type="checkbox" class="bn-triple-toggle"{_checked_attr("FORBIDDEN_USER")} data-cats="FORBIDDEN_USER"/> Forbidden V (User)
    </label>
    <label style="cursor:pointer;background:#ffe4e6;color:#e11d48;border:1px solid #e11d48;padding:2px 6px;border-radius:4px;">
      <input type="checkbox" class="bn-triple-toggle"{_checked_attr("INDUCED_FORBIDDEN_V")} data-cats="INDUCED_FORBIDDEN_V"/> Induced Forbidden V
    </label>
    <label style="cursor:pointer;background:#dcfce7;color:#16a34a;border:1px solid #16a34a;padding:2px 6px;border-radius:4px;">
      <input type="checkbox" class="bn-triple-toggle"{_checked_attr("ACTIVE_V_USER")} data-cats="ACTIVE_V_USER"/> Active V (User)
    </label>
    <label style="cursor:pointer;background:#ffedd5;color:#ea580c;border:1px solid #ea580c;padding:2px 6px;border-radius:4px;">
      <input type="checkbox" class="bn-triple-toggle"{_checked_attr("ACTIVE_V_INDUCED")} data-cats="ACTIVE_V_INDUCED"/> Forced V (Cannot Forbid - Cycle)
    </label>
    <label style="cursor:pointer;background:#ede9fe;color:#7c3aed;border:1px solid #7c3aed;padding:2px 6px;border-radius:4px;">
      <input type="checkbox" class="bn-triple-toggle"{_checked_attr("ACTIVE_V_COMPLETION")} data-cats="ACTIVE_V_COMPLETION"/> Active V (Maximal Completion)
    </label>
    <label style="cursor:pointer;background:#f1f5f9;color:#64748b;border:1px solid #94a3b8;padding:2px 6px;border-radius:4px;">
      <input type="checkbox" class="bn-triple-toggle"{_checked_attr("PREVENTED_NON_COLLIDER")} data-cats="PREVENTED_NON_COLLIDER"/> Non-Collider
    </label>
    <label style="cursor:pointer;background:#f8fafc;color:#94a3b8;border:1px solid #cbd5e1;padding:2px 6px;border-radius:4px;">
      <input type="checkbox" class="bn-triple-toggle"{_checked_attr("SHIELDED_TRIANGLE")} data-cats="SHIELDED_TRIANGLE"/> Shielded (Triangle)
    </label>
  </div>
  {raw_html}
  <details open style="margin-top:10px;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;padding:8px 10px;">
    <summary style="font-weight:600;font-size:13px;cursor:pointer;">
      User-Chosen Edge Orientations ({len(self.user_oriented)} chosen)
    </summary>
    <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:12.5px;text-align:left;">
      <thead>
        <tr style="background:#f1f5f9;">
          <th style="padding:6px 8px;border-bottom:2px solid #cbd5e1;width:48px;">#</th>
          <th style="padding:6px 8px;border-bottom:2px solid #cbd5e1;">Chosen Orientation</th>
          <th style="padding:6px 8px;border-bottom:2px solid #cbd5e1;">Skeleton Edge</th>
          <th style="padding:6px 8px;border-bottom:2px solid #cbd5e1;">Gadget / Scope</th>
        </tr>
      </thead>
      <tbody>
        {''.join(user_choice_rows)}
      </tbody>
    </table>
  </details>
  <details open style="margin-top:10px;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;padding:8px 10px;">
    <summary style="font-weight:600;font-size:13px;cursor:pointer;">
      Induced by Meek / Cycle Edge Orientations ({len(self.induced_oriented)} induced)
    </summary>
    <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:12.5px;text-align:left;">
      <thead>
        <tr style="background:#f1f5f9;">
          <th style="padding:6px 8px;border-bottom:2px solid #cbd5e1;width:48px;">#</th>
          <th style="padding:6px 8px;border-bottom:2px solid #cbd5e1;">Induced Orientation</th>
          <th style="padding:6px 8px;border-bottom:2px solid #cbd5e1;">Skeleton Edge</th>
          <th style="padding:6px 8px;border-bottom:2px solid #cbd5e1;">Rule / Trigger</th>
          <th style="padding:6px 8px;border-bottom:2px solid #cbd5e1;">Gadget / Scope</th>
          <th style="padding:6px 8px;border-bottom:2px solid #cbd5e1;">Explanation (Culprit Nodes &amp; Rule)</th>
        </tr>
      </thead>
      <tbody>
        {''.join(induced_rows)}
      </tbody>
    </table>
  </details>
  <details style="margin-top:10px;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;padding:8px 10px;">
    <summary style="font-weight:600;font-size:13px;cursor:pointer;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
      <span>Incident Edge Pairs &amp; V-Structure Constraints Status (<span class="bn-triples-count">{num_summary_triples} triples</span>)</span>
      <label onclick="event.stopPropagation();" style="font-weight:500;font-size:12px;color:#334155;cursor:pointer;display:inline-flex;align-items:center;gap:5px;background:#f1f5f9;padding:2px 8px;border-radius:4px;border:1px solid #cbd5e1;">
        <input type="checkbox" class="bn-sync-summary-toggle" style="cursor:pointer;"/>
        Filter table rows using the V-Structure / Triple checkboxes above
      </label>
    </summary>
    <div style="font-size:11.5px;color:#64748b;margin-top:6px;">
      Click any column header to sort ascending/descending. Triples of status <i>Cannot Forbid (Already Non-Collider)</i> are omitted from this summary.
    </div>
    <table class="bn-triples-table" style="width:100%;border-collapse:collapse;margin-top:6px;font-size:12.5px;text-align:left;">
      <thead>
        <tr style="background:#f1f5f9;user-select:none;">
          <th data-sort-col="0" style="padding:6px 8px;border-bottom:2px solid #cbd5e1;cursor:pointer;" title="Click to sort by Triple">Triple (u, w, v) <span class="bn-sort-indicator" style="font-size:11px;color:#64748b;">⇅</span></th>
          <th data-sort-col="1" style="padding:6px 8px;border-bottom:2px solid #cbd5e1;cursor:pointer;" title="Click to sort by Incident Edges">Incident Edges <span class="bn-sort-indicator" style="font-size:11px;color:#64748b;">⇅</span></th>
          <th data-sort-col="2" style="padding:6px 8px;border-bottom:2px solid #cbd5e1;cursor:pointer;" title="Click to sort by Status">Status <span class="bn-sort-indicator" style="font-size:11px;color:#64748b;">⇅</span></th>
          <th data-sort-col="3" style="padding:6px 8px;border-bottom:2px solid #cbd5e1;cursor:pointer;" title="Click to sort by Can Forbid V?">Can Forbid V? <span class="bn-sort-indicator" style="font-size:11px;color:#64748b;">⇅</span></th>
          <th data-sort-col="4" style="padding:6px 8px;border-bottom:2px solid #cbd5e1;cursor:pointer;" title="Click to sort by Explanation">Explanation <span class="bn-sort-indicator" style="font-size:11px;color:#64748b;">⇅</span></th>
        </tr>
      </thead>
      <tbody>
        {''.join(triple_rows)}
      </tbody>
    </table>
  </details>
</div>
"""
        return legend_and_tables

    def show(
        self,
        title: str = "Bayesian Network State",
        height: str = "540px",
        show_triple_badges: bool = True,
        show_shielded_badges: bool = True,
        save_html_path: Optional[str] = None,
        visible_triple_categories: Optional[Set[str]] = None,
    ) -> Any:
        """Display the interactive PyVis graph and status table inline in a Jupyter Notebook."""
        rendered = self.render_html(
            title=title,
            height=height,
            show_triple_badges=show_triple_badges,
            show_shielded_badges=show_shielded_badges,
            visible_triple_categories=visible_triple_categories,
        )
        if save_html_path:
            with open(save_html_path, "w", encoding="utf-8") as f:
                f.write(rendered)
        if _HAS_IPYTHON:
            return HTML(rendered)
        return rendered

    def print_status(self) -> None:
        """Print a readable text summary of gadgets, edges, induced derivations, and triples."""
        print(f"Nodes ({len(self.nodes)}): {sorted(self.nodes)}")
        if self.gadgets:
            print("Gadgets:")
            for gid, inst in self.gadgets.items():
                print(f"  [{gid}] template={inst.template.name}, center={inst.center}, nodes={list(inst.node_map.values())}")
        print("Edges:")
        for e in sorted(self.skeleton):
            u, v = e
            if e in self.user_oriented:
                s, d = self.user_oriented[e]
                print(f"  [USER]       {s} -> {d}")
            elif e in self.induced_oriented:
                s, d = self.induced_oriented[e]
                print(f"  [INDUCED]    {s} -> {d}   ({self.induced_reasons[e]})")
            elif e in self.completion_oriented:
                s, d = self.completion_oriented[e]
                role = self.completion_edge_roles.get(e, "COMPLETION")
                print(f"  [{role}] {s} -> {d}")
            else:
                print(f"  [UNDIRECTED] {u} -- {v}")
        print("\nIncident Edge Pairs (Triples, excluding Already Non-Collider):")
        for info in self.get_triples_summary(include_non_collider=False):
            t = info["triple"]
            print(
                f"  {t}: [{info['category']}] can_forbid={info['can_forbid']} | {info['explanation']}"
            )

def get_v_structs(graph):
    return {
            info["triple"]
            for info in graph.get_triples_summary()
            if info["category"] in ("ACTIVE_V_USER", "ACTIVE_V_INDUCED", "ACTIVE_V_COMPLETION")
        }
