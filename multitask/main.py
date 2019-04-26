"""
 Christian Roncal 
 CMSC 499 Dr. Leiserson
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall, Loss, RunningAverage
from ignite.contrib.metrics import AveragePrecision
from ignite.contrib.handlers import ProgressBar

from tqdm import tqdm
import argparse
from bmf_config import BMFConfig
from gmf_config import GMFConfig
from debug_gmf_config import GMFConfig_dbg

try:
    from tensorboardX import SummaryWriter
    print('gucci')
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


### Change config to switch between bmf and gmf



############################
## TensorboardX
############################
def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    vs, hs, ys = next(data_loader_iter)

    try:
        writer.add_graph(model, (vs, hs))
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


############################
## Ignite functions 
############################

def train_batch(engine, batch):    
    model.train()
    optimizer.zero_grad()
    vidxs, hidxs, ys = batch
    pred = model(vidxs, hidxs)
    loss = criterion(pred, ys)
    loss.backward()
    optimizer.step()
        
    return loss.item()

def eval_fn(engine, batch):
    model.eval()
    with torch.no_grad():
        vs, hs, ys = batch
        y_pred = model(vs, hs)
        return y_pred, ys

def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


"""
Runs the joint
"""
def run():

    ### Create tensorboardx writer
    writer = create_summary_writer(model, train_loader, log_dir)
    print("Writer created")

    ### Create trainer, evaulator
    trainer = Engine(train_batch)
    evaluator = Engine(eval_fn)

    #### Attach evaluation metrics
    Accuracy(output_transform=thresholded_output_transform).attach(evaluator, 'accuracy')
    Precision(output_transform=thresholded_output_transform).attach(evaluator, 'precision')
    Recall(output_transform=thresholded_output_transform).attach(evaluator, 'recall')
    AveragePrecision().attach(evaluator, 'AP')
    Loss(criterion).attach(evaluator, 'loss')

    #### tqdm settings
    if USE_TQDM:
            desc = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
            pbar = tqdm(
                    initial=0, leave=False, total=len(train_loader),
                    desc=desc.format(0, 0 , 0, 0)
            )

    ### Attach on events
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(train_loader) + 1
            if iter % log_interval == 0:
                    if USE_TQDM:
                            pbar.desc = desc.format(engine.state.epoch, iter, len(train_loader), engine.state.output)
                            pbar.update(log_interval)
                    else:
                            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                                    .format(engine.state.epoch, iter, len(train_loader), engine.state.output))
                    writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            prec = metrics['precision']
            recall = metrics['recall']
            ap = metrics['AP']
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Precision: {:.2f} Recall: {:.2f} APR: {:.2f}"
                    .format(engine.state.epoch, avg_accuracy, avg_loss, prec, recall, ap))
            writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
            writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)
            writer.add_scalar("training/precision", prec, engine.state.epoch)
            writer.add_scalar("training/recall", recall, engine.state.epoch)
            writer.add_scalar("training/avg precision", ap, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_val_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            prec = metrics['precision']
            recall = metrics['recall']
            ap = metrics['AP']
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Precision: {:.2f} Recall: {:.2f} APR: {:.2f}"
                    .format(engine.state.epoch, avg_accuracy, avg_loss, prec, recall, ap))
            writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)
            writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)
            writer.add_scalar("validation/precision", prec, engine.state.epoch)
            writer.add_scalar("validation/recall", recall, engine.state.epoch)
            writer.add_scalar("validation/avg precision", ap, engine.state.epoch)


    #### Run the joint
    print("Training...")
    trainer.run(train_loader, max_epochs=epochs)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPI Matrix Factorization")

    ## Train settings
    parser.add_argument('--model',  help="Choose between 'gmf' and 'bmf'")
    parser.add_argument('--bs', default=64, help="batch size", type=int)
    parser.add_argument('--epochs', default=15, help="epochs", type=int)
    parser.add_argument('--debug', dest="debug", action='store_true')
    parser.add_argument('--lr', default=.0001, help="learning rate", type=float)

    ## System settings
    parser.add_argument('--datapath', default='./data/', help="path where data be")
    parser.add_argument('--cpu', dest="use_cpu", action='store_true', help="Use cpu instead of cuda")

    ## tensorboard/ignite settings
    parser.add_argument('--logdir', default='./logs/', help="where tensorboard logs will be stored")
    parser.add_argument('--log_interval', default=10, help="log every x iterations", type=int)
    parser.add_argument('--no_tqdm', dest="no_tqdm", action='store_true')

    args = parser.parse_args()
    print(args)

    #### Training settings
    BS = args.bs
    device = 'cuda' if not args.use_cpu else 'cpu'
    epochs = args.epochs
    lr = args.lr

    #### System settings
    path = args.datapath
    USE_TQDM = not args.no_tqdm
    DEBUG = args.debug

    #### Tensorboard settings
    log_dir = args.logdir
    log_interval = args.log_interval

    if args.model == 'gmf':
        config = GMFConfig(path, DEBUG, device)
    elif args.model == 'bmf':
        config = BMFConfig(path, DEBUG, device)
    elif args.model == 'gmf_dbg':
        config = GMFConfig_dbg(device)
    else:
        print("Unrecognized model: ", args.model, ". pick betwen 'bmf' or 'gmf'.")

#     gen = config.get_generator()
#     train_loader = gen.create_train_loader(BS)
#     val_loader = gen.create_val_loader(BS)
#     test_loader = gen.create_test_loader(BS)
#     print('-' * 15, "Data loaders created", '-' * 15)

#     model = config.get_model()

#     ### Optimizer and loss
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.BCELoss()

#     print('-' * 15, "Optimizer and criterion", '-' * 15)
#     print(optimizer)
#     print()
#     print(criterion)
#     print('-' * 30)

#     run()

