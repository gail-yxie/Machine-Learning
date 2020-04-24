import torch
import torch.nn as nn
from torch import Tensor
from torchvision import datasets, transforms
import torch.nn.utils.weight_norm as wn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import copy
from tqdm import tqdm as pbar


class HyperParams(object):
    pass


def mkSavePath(key, p):
    if p.step_lr:
        path = "./{}/hs{}-ep{}-step{}".format(key, p.hidden_size, p.num_epochs, p.step_lr)
    else:
        path = "./{}/hs{}-ep{}".format(key, p.hidden_size, p.num_epochs)
    return path


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data(params):
    # Data loader parameters
    train_batch_size = 100
    test_batch_size = 1000
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    # Data loader
    params.train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True,
                                                                     transform=transforms.Compose(
                                                                         [transforms.ToTensor(),
                                                                          transforms.Normalize(
                                                                              (0.1307,),
                                                                              (0.3081,))])),
                                                      batch_size=train_batch_size, shuffle=True, **kwargs)

    params.test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                                                     batch_size=test_batch_size, shuffle=False, **kwargs)

    return params


# Define the key module template
class MyModule(nn.Module):

    def core_ops(self, out: Tensor) -> Tensor:
        pass

    def norm_ops(self, out: Tensor) -> Tensor:
        pass

    def __init__(self, params):
        super(MyModule, self).__init__()
        self.fc1 = nn.Linear(params.input_size, params.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(params.hidden_size, params.num_classes)
        self.__name__ = params.model_name
        self.opt = None
        self.H = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.norm_ops(out)
        out = self.relu(out)
        out = self.core_ops(out)
        out = self.fc2(out)
        return out


# Define subclasses of MyModule
class NoCoreModule(MyModule):
    def core_ops(self, out: Tensor) -> Tensor:
        return out


class NoNormModule(MyModule):
    def norm_ops(self, out: Tensor) -> Tensor:
        return out


class NeuralNet(NoNormModule):
    def __init__(self, params):
        super(NeuralNet, self).__init__(params)
        self.__net__ = "NeuralNet"

    def core_ops(self, out):
        self.H = torch.diag(1 / torch.norm(self.fc1.weight.detach(), p=2, dim=1))
        return out


class WNNet(NoCoreModule):
    def __init__(self, params):
        super(WNNet, self).__init__(params)
        self.fc1 = wn(self.fc1)  # add weight normalization here
        self.__net__ = "WNNet"

    def norm_ops(self, out):
        self.H = torch.diag(1 / torch.norm(self.fc1.weight_v.detach(), p=2, dim=1))
        return out


class BNNet(NoCoreModule):
    def __init__(self, params):
        super(BNNet, self).__init__(params)
        self.bn1 = nn.BatchNorm1d(params.hidden_size)
        self.__net__ = "BNNet"

    def norm_ops(self, out):
        self.H = torch.diag(1 / torch.norm(self.fc1.weight.detach(), p=2, dim=1))
        return self.bn1(out)


def build_model(name, Net, optimizer, params):
    params.model_name = name
    model = Net(params).to(device)
    model.opt = optimizer(model.parameters(), lr=params.learning_rate)
    return model


def train(model, params, print_log=False):
    model_name = "{}_lr{}".format(model.__name__, params.learning_rate)
    writer = SummaryWriter('{}/{}/'.format(mkSavePath('runs', params), model_name))
    optimizer = model.opt

    if params.step_lr:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=params.step_lr, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    best_model = copy.deepcopy(model.state_dict())
    best_ep = 0
    max_acc = test_model(model, params)
    save_H = False

    for epoch in pbar(range(params.num_epochs)):
        for phase in ['train', 'test']:
            logs = {'Loss': 0.0, 'Accuracy': 0.0}
            # Set the model to the correct phase
            model.train() if phase == 'train' else model.eval()

            for images, labels in getattr(params, phase + '_loader'):
                # Move tensors to the configured device
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    accuracy = torch.sum(torch.max(outputs, 1)[1] == labels.data).item()

                    # Update log
                    logs['Loss'] += images.shape[0] * loss.detach().item()
                    logs['Accuracy'] += accuracy

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if not save_H:
                        init_H = model.H.detach().cpu().numpy().diagonal()
                        max_H = None
                        save_H = True

            logs['Loss'] /= len(getattr(params, phase + '_loader').dataset)
            logs['Accuracy'] /= len(getattr(params, phase + '_loader').dataset)
            writer.add_scalars('Loss', {phase: logs['Loss']}, epoch+1)
            writer.add_scalars('Accuracy', {phase: logs['Accuracy']}, epoch+1)

            if print_log:
                print('\n Epoch [{}]: ({}) Loss = {:.6f}, Accuracy = {:.4f}%'
                      .format(epoch+1, phase, logs['Loss'], logs['Accuracy']*100))

            if phase == 'test' and logs['Accuracy'] > max_acc:
                max_acc = logs['Accuracy']
                best_ep = epoch + 1
                best_model = copy.deepcopy(model.state_dict())
                max_H = model.H.detach().cpu().numpy().diagonal()

        if params.step_lr:
            scheduler.step()

    # write to tensor board
    writer.add_text('Best_Accuracy', str(max_acc), best_ep)
    writer.add_histogram('init_H', init_H)
    writer.add_histogram('max_H', max_H, best_ep)

    # save model
    PATH = '{}/{}.pt'.format(mkSavePath('model', params), model_name)
    torch.save(best_model, PATH)

    writer.close()


def test_model(model, params, print_log=False):
    model = model.eval()
    phase = 'test'
    logs = {'Accuracy': 0.0}

    for images, labels in pbar(getattr(params, phase + '_loader')):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            accuracy = torch.sum(torch.max(outputs, 1)[1] == labels.data).item()
            logs['Accuracy'] += accuracy

    logs['Accuracy'] /= len(getattr(params, phase + '_loader').dataset)

    if print_log:
        print('Accuracy and Loss of the network on '
              'the 10000 test images: {}%'.format(accuracy))

    return logs['Accuracy']
