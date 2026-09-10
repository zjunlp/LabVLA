# LabVLA evaluation notes

The validated TransportBeaker evaluation uses `config/level3_TransportBeaker.yaml`, three RGB cameras, an eight-dimensional Franka state/action vector, and the OpenPI WebSocket client. Keep evaluator and training schema aligned: legacy LeRobot exports use `state`/`actions`, while canonical exports use `observation.state`/`action`.

```bash
python main.py --headless --no-video --config-name level3_TransportBeaker
```

Set `infer.host`, `infer.port`, and `max_episodes` in the configuration. Results are written under `outputs/infer/`; add `--save-action-trajectory` when action traces are needed.
