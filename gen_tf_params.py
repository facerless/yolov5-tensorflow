
import torch

pt_file=r'/home/chenrui/data/work/yolov5-master/runs/X800-FireSmoke-20201230/weights/best.pt'

device = torch.device("cpu")
model = torch.load(pt_file, map_location=device)
model=model['model'].float()  # load to FP32
model.to(device)
model.eval()

import pickle
data_dict={}
for k, v in model.state_dict().items():
    vr = v.cpu().numpy()
    data_dict[k]=vr
    
print(data_dict['model.24.anchors'])
print(data_dict['model.24.anchor_grid'])
fid=open('params.dict','wb')   
pickle.dump(data_dict,fid)
fid.close()