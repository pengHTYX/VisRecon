import os
from icecream import ic

root_folder = 'implicit/cubemaps'
cmgen_bin = '$HOME/filament/bin/cmgen'

# Generate sh coefficients for HDRs in `implicit/cubemaps`
for name in os.listdir(root_folder):
    print(name)
    folder_path = os.path.join(root_folder, name)
    hdr_path = os.path.join(folder_path, f"{name}.exr")
    output_path = os.path.join(folder_path, "sh.txt")
    cmd = f"{cmgen_bin}  --sh=3 --no-mirror --sh-output={output_path} {hdr_path}"
    os.system(cmd)
