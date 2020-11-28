import torch
torch.autograd.set_detect_anomaly(True)

x=torch.tensor([0,1,2],dtype=torch.float32,requires_grad=True)

a=x+2
b=a**2
c=b+3
y=c.mean()
print(y)
y.retain_grad()
y.backward()

k=0
