# Baseline Plugins

Drop a Python file in this directory and it will be auto-discovered at
startup and registered into the baseline algorithm dispatch.

## Plugin contract

Each `*.py` file must expose a top-level `register(registry)` function
that calls `registry["register"](short_code, display_name, compute_fn,
default_params)`.

The `compute_fn` signature is:

```python
def compute(x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """Return a baseline array the same length as y."""
```

## Example

See `rolling_ball.py` for a minimal working example.

## Using a plugin

Once registered, the plugin's `short_code` becomes available in the
baseline algorithm dispatch. You can invoke it via:

- `compute_baseline(y, short_code, params)`
- The interactive baseline corrector (if the UI supports generic
  algorithm selection)
- A JSON parameter file with `{"baseline": {"algorithm": short_code,
  "params": {...}}}`
