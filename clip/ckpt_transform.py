import mindspore as ms
import torch

def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu').state_dict()
    pt_params = []
    for name in par_dict:
        parameter = par_dict[name]
        if "ln_" in name:
            name=name.replace(".weight",".gamma").replace(".bias",".beta")
        elif name=='token_embedding.weight':
            name='token_embedding.embedding_table'
        pt_params.append({"name":name,"data":ms.Tensor(parameter.numpy())})
    return pt_params


pth_path = "./ViT-B-32.pt"
pt_param = pytorch_params(pth_path)
ms.save_checkpoint(pt_param, "./ViT-B-32.ckpt")

