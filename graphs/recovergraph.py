import torch

# Load the checkpoint
ckpt = torch.load("/mnt/weatherloss/WindPower/training/CI/GraphTransformer/checkpoint/GTCIFINAL/last.ckpt", map_location="cpu", weights_only=False)

# The graph is stored under the model's hyper_parameters
graph_data = ckpt["hyper_parameters"]["graph_data"]

# Save it as a standalone .pt file
torch.save(graph_data, "CI_graph.pt")