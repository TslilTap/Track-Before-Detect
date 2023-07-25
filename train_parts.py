import torch
import torch.nn.functional as F
import datetime
from unet_model import UNet
from torch.utils.data import DataLoader


import os
import torch.optim as optim
import time

def train_model(model,
                 train_loader,
                 valid_loader,
                 learning_rate=0.001,
                 epochs=3000,
                 epochs_0=50,
                 weight_decay: float = 1,
                 checkpoint_path=None):

    train_loss_list = list()
    train_acc_list = list()
    valid_loss_list = list()
    valid_acc_list = list()



    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay,
                           betas=(0.90, 0.999))



    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        # Train
        model.train()
        for i, (input, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(input)

            acc = get_accuracy(output, label)
            train_acc += acc

            if epoch < epochs_0:
                loss = pixel_BCE_loss2(output, label, weight=0.8)
            else:
                loss = pixel_BCE_loss(output, label)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        with torch.no_grad():
            model.eval()
            for i, (input, label) in enumerate(valid_loader):
                output = model(input)

                loss = pixel_BCE_loss(output, label)
                valid_loss += loss.item()
                valid_acc += get_accuracy(output, label)

        # Print statistics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        epoch_time = time.time() - start_time
        remaining_time = epoch_time * (epochs - epoch - 1)

        formatted_epoch_time = format_time(epoch_time)
        formatted_remaining_time = format_time(remaining_time)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc * 100:.2f}%")
        print(f"Time taken: {formatted_epoch_time}, Estimated remaining time: {formatted_remaining_time}")

        if checkpoint_path is not None:
            torch.save(model.state_dict(), checkpoint_path)

    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list


def pixel_BCE_loss(input, target,weight = 0.5):
    batch_size, NrNd = input.size()

    pos = (NrNd - 1) * (1 - weight)
    neg = 1*weight


    # Calculate pos_weight and neg_weight for each element in the batch
    for i in range(batch_size):
        label = torch.zeros(NrNd)
        label[target[i]] = 1
        pos_weight = torch.ones(NrNd) * neg
        pos_weight[target[i]] = pos

        # Compute the binary cross-entropy loss for each element in the batch
        bce_loss = F.binary_cross_entropy_with_logits(input[i].view(-1), label.float(), pos_weight=pos_weight)
        if i == 0:
            loss = bce_loss
        else:
            loss += bce_loss

    loss /= batch_size
    return loss

def pixel_BCE_loss2(input, target,weight = 0.5):
    batch_size, NrNd = input.size()

    input_flat = input.view(batch_size, -1)


    # Compute binary cross-entropy loss
    pos = (NrNd - 1)
    neg = 1
    # Calculate pos_weight and neg_weight for each element in the batch
    for i in range(batch_size):
        label = torch.zeros(NrNd)
        label[target[i]] = 1
        pos_weight = torch.ones(NrNd) * neg
        pos_weight[target[i]] = pos

        # Compute the binary cross-entropy loss for each element in the batch
        bce_loss = F.binary_cross_entropy_with_logits(input[i].view(-1), label.float(), pos_weight=pos_weight)
        if i == 0:
            BCE_loss = bce_loss
        else:
            BCE_loss += bce_loss

    BCE_loss /= batch_size

    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(input_flat, target) / 10

    # Combine the losses based on the provided weight
    loss = (BCE_loss * weight) + (ce_loss * (1 - weight))

    return loss


def get_accuracy(input, target):
    """
    Compute the accuracy of the model's predictions on a batch of inputs.
    """
    batch_size, _, = input.size()

    # compute the argmax of the predicted logits for each pixel
    pred_flat = torch.argmax(input.view(batch_size, -1), dim=1)

    # compute the number of correct predictions
    correct = sum([1 for i in range(batch_size) if pred_flat[i] == (target[i])])

    # compute the accuracy as a percentage
    accuracy = correct / batch_size

    return accuracy



def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))
