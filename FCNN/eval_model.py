import torch
import torch.nn as nn
import Model

model = Model.FCNN()
PATH = "model_params\model_state_dict"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
Model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_save = checkpoint['epoch']
loss = checkpoint['loss']


def test(loss):
    accuracy = 0
    count=0
    print(f"Evaluating model with epoch{epoch_save} and loss{loss}")
    model.eval()
    avg_loss = 0
    for images, targets in Model.test_dataloader:
        count+=1
        with torch.no_grad():
            output = model(images)
            loss = Model.loss_fn(output, targets)
            avg_loss += loss
            y = nn.Sigmoid()
            if(targets == int(y(output)+0.5)):
                accuracy += 1
            print(f"Output: {y(output)}, Target: {targets}")
            print(f"Avf Loss:{avg_loss/count} , Accuracy = {accuracy/count}")

test(loss)
