# Wind Turbine Performance Modeling

Modeling and analysis of a 2.5 MW, 3-blade, variable-speed, pitch-controlled wind turbine using a Blade Element–Momentum (BEM) approach and a tapered tower structural model. The project reproduces required deliverables (CP/CT single point, pitch sweep, λ–β map, power-cap pitch schedule, and tower bending/deflection) and auto-exports figures/tables.

---

## Features
- Single-entry MATLAB script (`main.m`) with local helpers.
- BEM kernel with Prandtl tip loss and high-induction correction.
- Deliverables:
  - **D1**: Single-point CP/CT @ (V=10 m/s, rpm=14, β=0°)
  - **D2**: CP vs β at fixed TSR (find β*)
  - **D3**: CP(λ,β) map at fixed V
  - **D4**: Pitch schedule β(V) to cap at 2.5 MW (rpm\_max)
  - **D5**: Tower stress & top deflection under thrust from D4

---

## Repo Layout 

```
wind-turbine-me4053/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ .gitignore
├─ main.m
├─ main.prj
├─ data/
│  ├─ blade geometry.csv
│  ├─ airfoil_DU91.csv
│  ├─ airfoil_DU93.csv
│  ├─ airfoil_DU96.csv
│  ├─ airfoil_DU97.csv
│  └─ tower specs.csv
├─ outputs/              # generated; stays empty in Git
│  └─ (plots, .csv)
└─ docs/
   ├─ figures/           # copies of final figures for the report
   └─ report-notes.md    # optional: outline, figure captions

