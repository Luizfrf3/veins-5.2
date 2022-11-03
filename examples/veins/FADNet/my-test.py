import torch
import torchvision.transforms as transforms
from utils.utils import get_model
from loaders.driving import get_iterator_driving

name = 'driving'
model = 'FADNet_plus'
device = 'cpu'
optimizer = 'adam'
lr_scheduler = 'constant'
initial_lr = 1e-4
epoch_size = 1
n_epochs = 20
path = 'data/data.npz'

model = get_model(
    name, model, device, optimizer_name=optimizer, lr_scheduler=lr_scheduler,
    initial_lr=initial_lr, epoch_size=epoch_size
)

print(model.net)
print(type(model.net.state_dict()))
print('Parameters: ', sum(param.numel() for param in model.net.parameters()))
#import pickle
#b = pickle.dumps(model.net.state_dict())
#s = b.decode('latin-1')
#print(type(pickle.loads(s.encode('latin-1'))))

train_iterator = get_iterator_driving(path, device)

print(train_iterator)

model.fit_iterator(train_iterator=train_iterator, n_epochs=n_epochs, verbose=1)

transform = transforms.ToTensor()
with torch.no_grad():
    for x, y in train_iterator:
        x = x.to(device)
        y = y.unsqueeze(-1).to(device)
        prediction = model.net(x)
        print(prediction, y)