import yaml

# update with your local configuration
base_folder = "/Users/Zian/Trabajo/IIMAS/SCC-Trees/data/db_Elena/ArbolesNorm/"
scc_folder = base_folder + "SSC/"
data = {
    "base_folder": base_folder,
    "binary_image": "Norm3_V_sk2.tif", 
    "start_position": {"x":154, "y":381},
    "output_file": "Norm3_V_sk(1).scc",
    "distances_file": "Norm3_V_sk(1)_d.scc"
}

yaml_output = yaml.dump(data, sort_keys=False)
with open('./config.yaml', 'w') as file:
    file.write(yaml_output)