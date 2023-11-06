The folder contains scripts to generate teaser folder

Most files (`target_python_file.py`) are runnable with

```
python -m teaser.target_python_file --data_folder /path/to/input/folder --out /path/to/output/folder
```

where `/path/to/input/folder` contains a list of subfolders for output (i.e. `out/vis_fuse/test/4``), each of which must contain 'prt_gen.npy' file (with `--save` passed in output generation)
