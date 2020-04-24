import os
import utils
from utils import *


def experiment(models_exp, optims_exp, lrs=False, hs=500, ep=5, step_lr=False):
    # set hyper parameters
    params = HyperParams()
    params.input_size = 784
    params.hidden_size = hs
    params.num_classes = 10
    params.num_epochs = ep
    params.step_lr = step_lr  # Period of learning rate decay

    # set dictionaries
    models = {"nmodel": utils.NeuralNet, "wmodel": utils.WNNet, "bmodel": utils.BNNet}
    optims = {"sgd": torch.optim.SGD, "adagrad": torch.optim.Adagrad, "adam": torch.optim.Adam}
    if not lrs:
        lrs = [0.1, 0.01, 0.001, 0.0001]

    # create path to save records
    # path = mkSavePath('record', params)
    model_path = mkSavePath('model', params)
    board_path = mkSavePath('runs', params)
    # if not os.path.exists(path):
    #     os.mkdir(path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # build data loader
    torch.manual_seed(1023)
    params = get_data(params)

    def run_model(model_name, Net, opt, params, print_log=False):
        model = build_model(model_name, Net, opt, params)
        train(model, params, print_log)

    # run a batch of models
    for model in models_exp:
        for optim in optims_exp:
            model_r = "{}_{}".format(model, optim)
            for lr in lrs:
                if os.path.exists("{}/{}_lr{}/".format(board_path, model_r, lr)):
                    print("The record of the model: {}_lr{} already exists!".format(model_r, lr))
                    continue
                else:
                    print("Start training {} with lr={} ...".format(model_r, lr))
                    params.learning_rate = lr
                    run_model(model_r, models[model], optims[optim], params, True)
                    print("Finish training {} with lr={} ...".format(model_r, lr))
