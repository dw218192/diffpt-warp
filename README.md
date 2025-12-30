# Differentiable rendering via path replay for material optimization
This is a Hello-World project for the purpose of learning the basics of differentiable rendering and programming in general.



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

By default, the render output will be saved as raw HDR to `./_output/target/cornell_sphere_target.hdr`.

To save an 8-bit tonemapped PNG instead, add `--save-png`.
