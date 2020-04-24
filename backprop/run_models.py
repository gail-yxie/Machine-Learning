from experiment import experiment
models_exp = ['nmodel', 'wmodel', 'bmodel'] # choose from: ['nmodel', 'wmodel', 'bmodel']
optims_exp = ['adam'] # choose from: ['sgd', 'adagrad', 'adam']
lrs = [0.01] # choose from: lrs = [0.1, 0.01, 0.001, 0.0001]
experiment(models_exp, optims_exp, lrs, hs=500, ep=100, step_lr=20)
# you can change hs: hidden layer size; ep: epochs; step_lr: steps to decay learning rate
