import torch
import torch.nn as nn

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Precision, Recall, Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar

### Change config to switch between bmf and gmf
from bmf_config import *

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

trainer = Engine(train_batch)
train_evaluator = Engine(eval_fn)

### Eval Metrics
def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y

Accuracy(output_transform=thresholded_output_transform).attach(train_evaluator, 'accuracy')
Precision(output_transform=thresholded_output_transform).attach(train_evaluator, 'precision')
Recall(output_transform=thresholded_output_transform).attach(train_evaluator, 'recall')
Loss(criterion).attach(train_evaluator, 'loss')

### Attach on events
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    accuracy = metrics['accuracy']
    prec = metrics['precision']
    rec = metrics['recall']
    pbar.log_message(
        "\n Training Results - Epoch: {}  Acc: {:.2f} Prec: {:.2f} Rec: {:.2f}"
        .format(engine.state.epoch, accuracy, prec, rec))


RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
pbar = ProgressBar(persist=True)

print("Training...")
trainer.run(train_loader, max_epochs=n_epochs)