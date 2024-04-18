import anndata as ad
import pandas as pd
import subprocess
from os import path
import yaml
import re

## VIASH START
meta = {
    "executable": "target/docker/denoising/methods/dca/dca",
    "config": "target/docker/denoising/methods/dca/.config.vsh.yaml",
    "resources_dir": "resources_test/denoising"
}
## VIASH END

# helper functions
def check_h5ad_slots(adata, arg):
    """Check whether an AnnData file contains all for the required
    slots in the corresponding .info.slots field.
    """
    for struc_name, items in arg["info"].get("slots", {}).items():
        struc_x = getattr(adata, struc_name)
        
        if struc_name == "X":
            if items.get("required", True):
                assert struc_x is not None,\
                    f"File '{arg['value']}' is missing slot .{struc_name}"
        
        else:
            for item in items:
                if item.get("required", True):
                    assert item["name"] in struc_x,\
                        f"File '{arg['value']}' is missing slot .{struc_name}['{item['name']}']"

def check_df_columns(df, arg):
    """Check whether a DataFrame contains all for the required
    columns in the corresponding .info.columns field.
    """
    for item in arg["info"].get("columns", []):
        if item.get("required", True):
            assert item['name'] in df.columns,\
                f"File '{arg['value']}' is missing column '{item['name']}'"

def run_and_check_outputs(arguments, cmd):
    print(">> Checking whether input files exist", flush=True)
    for arg in arguments:
        if arg["type"] == "file" and arg["direction"] == "input":
            assert path.exists(arg["value"]), f"Input file '{arg['value']}' does not exist"

    print(f">> Running script as test", flush=True)
    out = subprocess.run(cmd, stderr=subprocess.STDOUT)

    if out.stdout:
        print(out.stdout)

    if out.returncode:
        print(f"script: \'{' '.join(cmd)}\' exited with an error.")
        exit(out.returncode)

    print(">> Checking whether output file exists", flush=True)
    for arg in arguments:
        if arg["type"] == "file" and arg["direction"] == "output":
            assert path.exists(arg["value"]), f"Output file '{arg['value']}' does not exist"

    print(">> Reading h5ad files and checking formats", flush=True)
    for arg in arguments:
        file_type = arg.get("info", {}).get("file_type", "h5ad")
        if arg["type"] == "file":
            if file_type == "h5ad" and "slots" in arg["info"]:
                print(f"Reading and checking {arg['clean_name']}", flush=True)

                # try to read as an anndata, else as a parquet file
                adata = ad.read_h5ad(arg["value"])

                print(f"  {adata}")

                check_h5ad_slots(adata, arg)
            elif file_type in ["parquet", "csv"] and "columns" in arg["info"]:
                print(f"Reading and checking {arg['clean_name']}", flush=True)

                if file_type == "csv":
                    df = pd.read_csv(arg["value"])
                else:
                    df = pd.read_parquet(arg["value"])
                print(f"  {df}")
                
                check_df_columns(df, arg)


    print("All checks succeeded!", flush=True)


# read viash config
with open(meta["config"], "r") as file:
    config = yaml.safe_load(file)

# get resources
arguments = []

for arg in config["functionality"]["arguments"]:
    new_arg = arg.copy()

    # set clean name
    clean_name = re.sub("^--", "", arg["name"])
    new_arg["clean_name"] = clean_name

    # use example to find test resource file
    if arg["type"] == "file":
      if arg["direction"] == "input":
          value = f"{meta['resources_dir']}/{arg['example'][0]}"
      else:
          value = f"{clean_name}.h5ad"
      new_arg["value"] = value
    
    arguments.append(new_arg)


if "test_setup" not in config["functionality"]["info"]:
    argument_sets = {"run": arguments}
else:
    test_setup = config["functionality"]["info"]["test_setup"]
    argument_sets = {}
    for name, test_instance in test_setup.items():
        new_arguments = []
        for arg in arguments:
            new_arg = arg.copy()
            if arg["clean_name"] in test_instance:
                val = test_instance[arg["clean_name"]]
                if new_arg["type"] == "file" and new_arg["direction"] == "input":
                    val = f"{meta['resources_dir']}/{val}"
                new_arg["value"] = val
            new_arguments.append(new_arg)
        argument_sets[name] = new_arguments

for argset_name, argset_args in argument_sets.items():
    print(f">> Running test '{argset_name}'", flush=True)
    # construct command
    cmd = [ meta["executable"] ]
    for arg in argset_args:
        if arg["type"] == "file":
            cmd.extend([arg["name"], arg["value"]])

    run_and_check_outputs(argset_args, cmd)