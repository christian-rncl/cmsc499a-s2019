import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall, Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar

from tqdm import tqdm

try:
    from tensorboardX import SummaryWriter
    print('gucci')
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


### Change config to switch between bmf and gmf
# from bmf_config import *
from gmf_config import *

############################
## Optimizer and loss settings
############################
optimizer = torch.optim.SGD(model.parameters(), 
                            lr = 1e-3,  
                            momentum=0.9, 
                            weight_decay=1e-5)

criterion = nn.BCELoss()
# criterion = nn.MSELoss()

print('-' * 15, "Optimizer and criterion", '-' * 15)
print(optimizer)
print()
print(criterion)
print('-' * 30)

############################
## TensorboardX
############################
def create_summary_writer(model, data_loader, log_dir):
#     print("aye")
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    vs, hs, ys = next(data_loader_iter)
#     print('vs, hs: ', vs, hs)
    try:
        writer.add_graph(model, (vs, hs))
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

writer = create_summary_writer(model, train_loader, log_dir)
print("Writer created")

############################
## Train engine
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

trainer = Engine(train_batch)
# trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

### Evaluator 
# evaluator = create_supervised_evaluator(model, 
#                                         metrics = {'accuracy': Accuracy(),
#                                                    'loss': Loss(criterion),
#                                                    'precision': Precision(output_transform=thresholded_output_transform),
#                                                    'recall': Recall(output_transform=thresholded_output_transform)
#                                                    },
#                                                     device=device)
evaluator = Engine(eval_fn)

Accuracy(output_transform=thresholded_output_transform).attach(evaluator, 'accuracy')
Precision(output_transform=thresholded_output_transform).attach(evaluator, 'precision')
Recall(output_transform=thresholded_output_transform).attach(evaluator, 'recall')
Loss(criterion).attach(evaluator, 'loss')

## tqdm
if USE_TQDM:
        desc = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
        pbar = tqdm(
                initial=0, leave=False, total=len(train_loader),
                desc=desc.format(0, 0 , 0, 0)
        )

### Eval Metrics

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
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Precision: {:.2f} Recall: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_loss, prec, recall))
        writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)
        writer.add_scalar("training/precision", prec, engine.state.epoch)
        writer.add_scalar("training/recall", recall, engine.state.epoch)

@trainer.on(Events.EPOCH_COMPLETED)
def log_val_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        prec = metrics['precision']
        recall = metrics['recall']
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Precision: {:.2f} Recall: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_loss, prec, recall))
        writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)
        writer.add_scalar("validation/precision", prec, engine.state.epoch)
        writer.add_scalar("validation/recall", recall, engine.state.epoch)



print("Training...")
trainer.run(train_loader, max_epochs=epochs)
writer.close()


### OLD IGNITE

### Attach on events
# @trainer.on(Events.EPOCH_COMPLETED)
# def log_training_results(engine):
#     train_evaluator.run(train_loader)
#     metrics = train_evaluator.state.metrics
#     accuracy = metrics['accuracy']
#     prec = metrics['precision']
#     rec = metrics['recall']
#     pbar.log_message(
#         "\n Training Results - Epoch: {}  Acc: {:.2f} Prec: {:.2f} Rec: {:.2f}"
#         .format(engine.state.epoch, accuracy, prec, rec))


# RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
# pbar = ProgressBar(persist=True)