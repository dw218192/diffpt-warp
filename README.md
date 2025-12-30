# Differentiable rendering via path replay for material optimization

## Setup and Running the Renderer

Simply run the following command to initialize the project (if needed) and see the available options.

```bash
uv run ./src/main.py -h
```

### Example Usage
To render a simple stage with 1000 samples per pixel,
```bash
uv run ./src/main.py --spp 1000 --usd-path ./stages/cornell_sphere_target.usda --save-path ./_output/target
```

The render output will be saved to `./_output/target/cornell_sphere_target.png`.
