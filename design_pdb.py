import torch
from abflow.structure import write_to_pdb
from train import setup_model
from utils.load_data import Loader
from utils.utils import load_config
from abflow.structure import write_to_pdb
import os
import numpy as np
import torch

model_name = "predict_backbone_HCDR3_200e"
seed = 42
num_structures = 1
device = torch.device("cuda:0")
batch_size = 1

checkpoint_path = f"/scratch/hz362/datavol/model/{model_name}/epoch=199.ckpt"
load_optimizer = False
config_path = f"/scratch/hz362/datavol/model/{model_name}/config.yaml"

# load model
config = load_config(config_path)
config["num_workers"] = 1
config["batch_size"] = batch_size
config["devices"] = 1
design_mode = config["design_mode"]
model = setup_model(
    config, checkpoint_path=checkpoint_path, load_optimizer=load_optimizer
)

# redesign complex
ds_loader = Loader(config, dataset_name=config["dataset"])

model.eval()
model.to(device)

data_iter = iter(ds_loader.test_loader)
template_complex = next(data_iter)
# need to extract pdb, prepare cropping, centreing, and padding for the model generate class


redesign_complexes, redesign_trajs = model.generate(
    template_complex, design_mode=design_mode, seed=seed
)

des_dir = f"/home/jovyan/flow-matching-datavol/model/{model_name}/desabs"
if not os.path.exists(des_dir):
    os.makedirs(des_dir)
write_to_pdb(
    data=template_complex["full_complex"], filepath=f"{des_dir}/full_complex.pdb"
)
print("true id is", template_complex["full_complex"]["id"])
write_to_pdb(data=template_complex, filepath=f"{des_dir}/WT.pdb")
write_to_pdb(data=redesign_complexes, filepath=f"{des_dir}/DesAb.pdb")
# write traj to pdb - remember plddt per residue is also in redesign_complexes.
for t, des_t in enumerate(redesign_trajs):
    write_to_pdb(data=des_t, filepath=f"{des_dir}/DesAb_{t}.pdb")
