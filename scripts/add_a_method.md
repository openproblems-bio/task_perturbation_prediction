# How to add a method

## Prerequisites

Make sure you have Viash and Docker installed. If not, follow [these instructions](https://openproblems.bio/documentation/fundamentals/requirements) to get everything set up.

## Steps

### 1. Sync the test data

```bash
scripts/download_resources.sh
```

### 2. Create a new method component

Make sure to replace `foo` with the name of your method. Optionally, you can replace `python` with `r` if you are creating an R method.

```bash
common/create_component/create_component \
  --task perturbation_prediction \
  --language "python" \
  --name "foo" \
  --api_file src/api/comp_method.yaml \
  --output "src/methods/foo"
```

### 3. Fill in metadata

Edit the metadata file at `src/methods/foo/config.vsh.yaml` to describe your method.

### 4. Implement the method

Implement your method in the file `src/methods/foo/script.py` (or `src/methods/foo/script.R` if you are using R).

### 5. Test the component

```bash
viash test src/methods/foo/config.vsh.yaml
```
