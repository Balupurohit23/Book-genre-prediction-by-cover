import torch


def train_epoch(model, loss_fn, optimizer, train_loader, pbar_train, history, device):
    model.train()
    torch.set_grad_enabled(True)

    # Feeding batches
    for idx, data in enumerate(train_loader):
        out = model(data[0].to(device))
        labels = data[1].to(device)

        optimizer.zero_grad()
        loss = loss_fn(out, labels)

        loss.backward()
        optimizer.step()

        acc = get_acc(out, labels)

        log = [loss.item(), acc.item()]

        history.update_batch(log, mode='training')
        pbar_train.set_batch_postfix(history.train_batch_logs, mode='training')

        pbar_train.update(data[0].size()[0])
    return 0


def validate(model, val_loader, loss_fn, pbar_val, history, device):
    for idx, data in enumerate(val_loader):
        out = model(data[0].to(device))
        labels = data[1].to(device)

        loss = loss_fn(out, labels)

        acc = get_acc(out, labels)

        log = [loss.item(), acc.item()]

        history.update_batch(log, mode='validation')
        pbar_val.set_batch_postfix(history.val_batch_logs, mode='validation')
        pbar_val.update(data[0].size()[0])
    return 0


def get_acc(out, label):
    _, prediction = torch.max(out, 1)

    return torch.sum(prediction == label, dtype=torch.float32) / label.size()[-1]
