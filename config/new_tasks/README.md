# Additional LabUtopia tasks

This directory contains task configurations built from additional assets.

## Task catalog

| Level | Configuration | Task | Controller | Assets |
|---|---|---|---|---|
| 1 | `level1_pipette_pick.yaml` | Pick a pipette | `pipette_pick` | Pipette, PipetteStand |
| 1 | `level1_testtube_pick.yaml` | Pick a test tube | `testtube_pick` | TestTube, TubeRack |
| 2 | `level2_centrifuge_load.yaml` | Load a centrifuge | `centrifuge_load` | Centrifuge, TestTube |
| 2 | `level2_scale_measure.yaml` | Weigh an object | `scale_measure` | ElectronicScale, Beaker |
| 3 | `level3_incubator_load.yaml` | Load an incubator | `incubator_load` | Incubator, Petri_Dish |
| 3 | `level3_rotavap_setup.yaml` | Set up a rotary evaporator | `rotavap_setup` | Rotavapor, Round_bottomFlask |
| 4 | `level4_fumehood_operation.yaml` | Operate a fume hood | `fumehood_operation` | FumeHood, ErlenmeyerFlask |
| 4 | `level4_microscope_sample.yaml` | Prepare a microscope sample | `microscope_sample` | Microscope, Dropper, GlassSlide |
| 5 | `level5_full_centrifuge.yaml` | Run a full centrifuge workflow | `full_centrifuge` | Centrifuge, TestTube, TubeRack |
| 5 | `level5_chemical_synthesis.yaml` | Run a chemical synthesis workflow | `chemical_synthesis` | Flask, Condenser, HeatingMantle, Stirrer, Funnel |
| Additional | `new_scenario_spectrophotometer.yaml` | Operate a spectrophotometer | `spectrophotometer` | Spectrophotometer, Cuvette |
| Additional | `new_scenario_pcr_setup.yaml` | Prepare a PCR run | `pcr_setup` | PCR machine, PCR tube, Micropipette |
| Additional | `new_scenario_gel_electrophoresis.yaml` | Run gel electrophoresis | `gel_electrophoresis` | Electrophoresis apparatus, Gel tray |
| Additional | `new_scenario_ph_measurement.yaml` | Measure pH | `ph_measurement` | pH meter, Beaker |
| Additional | `new_scenario_autoclave_sterilization.yaml` | Run autoclave sterilization | `autoclave_sterilization` | Autoclave, Flask, Beaker, Tray |

The corresponding task and controller classes are registered in `tasks/` and
`controllers/` factories. Each configuration requires a matching USD scene and
may need task-specific physics or controller tuning. Success checks are defined
by the task/controller implementation.
