# Bayesian Network, V-Structure, Meek Rules & Gadget Explorer

An interactive Python toolkit (`v8.py`) for constructing and exploring **Bayesian network skeletons**, **forbidden v-structure constraints**, **Meek orientation rules ($R_1$–$R_4$) with cycle-avoidance lookahead**, **inclusion-maximal v-structure DAG completion**, and **prefabricated subgraph gadgets** (such as Variable and Glue gadgets), rendered interactively via `pyvis` / `vis-network`.

## Features

- **Undirected Skeleton & Constraint Propagation**:
  - Define arbitrary undirected graphs $G = (V, E)$, orient edges (`orient_edge(u, v)`), and forbid v-structures (`forbid_v_structure(u, w, v)`).
  - Automatically computes the closure under **Meek rules ($R_1$–$R_4$)**, **lookahead cycle avoidance**, and **global DAG backbone propagation**.
  - Reports detailed **culprit-node and shortest-path explanations** for every induced edge orientation, forced v-structure, and blocked orientation choice.
- **Inclusion-Maximal V-Structure Completion**:
  - Greedily extends the current partially directed graph into a valid DAG $A$ such that no other valid DAG $B$ respecting the same forbidden v-structures satisfies $\text{vstruct}(B) \supsetneq \text{vstruct}(A)$ (`complete_maximal_v_structures()`).
- **Prefabricated Subgraph Gadgets**:
  - Instantiate reusable subgraph templates (`GadgetTemplate.variable_gadget()`, `GadgetTemplate.pentagon_with_apex()`, or custom `GadgetTemplate` blueprints) via `add_gadget()`.
  - Connect multiple variable gadgets via `add_glue_gadget()` or arbitrary edges via `link_gadgets()`.
  - Move gadgets as rigid blocks in the interactive canvas (or toggle to single-node dragging).
- **Interactive HTML / Jupyter Visualization (`show()` / `render_html()`)**:
  - Synchronized annotation toggles (unchecked and hidden by default on the canvas; check any category to display its corner badges on the graph).
  - Collapsible tables for **User-Chosen Edge Orientations** and **Induced by Meek / Cycle Edge Orientations** (with rule badges and culprit traces).
  - **Sortable & Filterable Incident Edge Pairs Summary Table** (click any column header to sort; optionally sync row visibility with the category checkboxes above; automatically omits redundant `"Cannot Forbid (Already Non-Collider)"` triples).

## Installation

Python 3.8+ is required. The core engine runs using only the Python standard library and includes a built-in standalone `vis-network` HTML fallback, so no external packages are strictly required. For full Jupyter Notebook integration with a virtual environment:

```bash
python3 -m venv meek_engine

source meek_engine/bin/activate

pip install --upgrade pip

pip install pyvis jupyterlab

jupyter lab
```

## Quick Start

### 1. Exploring a Skeleton, Forbidden V-Structures & Meek Propagation

```python
from v8 import BayesianNetworkExplorer

# Create an induced 4-cycle A - B - C - D - A
exp = BayesianNetworkExplorer(edges=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])

# Forbid v-structures on three corners
exp.forbid_v_structure("D", "A", "B")
exp.forbid_v_structure("A", "B", "C")
exp.forbid_v_structure("B", "C", "D")

# Inspect induced orientations and culprit explanations in the terminal
exp.print_status()

# Render interactive HTML (displays inline in Jupyter or saves to a standalone HTML file)
exp.show(title="Induced 4-Cycle with 3 Forbidden Corners", save_html_path="c4_demo.html")
```

### 2. Using Variable and Glue Gadgets & Maximal V-Structure Completion

```python
from v8 import BayesianNetworkExplorer, GadgetTemplate

exp = BayesianNetworkExplorer()

# Instantiate two Variable Gadgets and connect them with a Glue Gadget
var_tpl = GadgetTemplate.variable_gadget()
g1 = exp.add_gadget(var_tpl, "G1")
g2 = exp.add_gadget(var_tpl, "G2")
exp.add_glue_gadget("g12", [g1, g2])

# Greedily complete the graph into an inclusion-maximal v-structure DAG
result = exp.complete_maximal_v_structures(seed=42)
print("Added V-structures:", result["added_v_structures"])

# Save or display the interactive visualization
exp.show(title="Glued Variable Gadgets (Maximal V-Structure DAG)", save_html_path="gadgets_demo.html")
```

## API Overview

| Method | Description |
| :--- | :--- |
| `add_node(u, pos=None)` | Add a node `u` with optional `(x, y)` coordinates |
| `add_skeleton_edge(u, v)` | Add an undirected edge `(u, v)` to the skeleton |
| `orient_edge(u, v)` | Orient `u -> v` and propagate Meek rules & cycle lookahead |
| `unorient_edge(u, v)` | Remove a user-chosen orientation on `(u, v)` |
| `forbid_v_structure(u, w, v)` | Forbid collider `u -> w <- v` (or pass two incident edges `((u, w), (v, w))`) |
| `unforbid_v_structure(u, w, v)` | Remove a user-specified forbidden v-structure constraint |
| `complete_maximal_v_structures(seed=None)` | Complete the PDAG into a DAG with an inclusion-maximal set of v-structures |
| `clear_completion()` | Remove edges added by `complete_maximal_v_structures()` |
| `add_gadget(template, instance_id, ...)` | Instantiate a `GadgetTemplate` under namespace `instance_id` |
| `add_glue_gadget(glue_id, var_gadgets)` | Connect $k \ge 2$ variable gadgets with a glue gadget |
| `show(title=..., save_html_path=...)` | Display interactive graph and status tables in Jupyter or save to HTML |
