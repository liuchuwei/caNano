import torch


def train(dataloader, model, loss_fn_1, loss_fn_2, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (S, X, y, r) in enumerate(dataloader):
        S, X, y, r = S.to(device), X.to(device), y.to(device), r.to(device)
        # y = torch.reshape(y, (-1, 1))
        y = torch.reshape(y, (-1, ))
        r = torch.reshape(r, (-1, 1))
        # Compute prediction error
        read_prob, site_ratio = model(S, X)
        loss1 = loss_fn_1(read_prob, y)
        loss2 = loss_fn_2(site_ratio, r)*10
        loss = loss1 + loss2
        # loss = loss1

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def evaluate(dataloader, model, loss_fn_1, loss_fn_2, device, type="val"):
    size = len(dataloader.dataset)*dataloader.dataset.min_reads
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for  S, X, y, r in dataloader:
            S, X, y, r = S.to(device), X.to(device), y.to(device), r.to(device)
            y = torch.reshape(y, (-1,))
            r = torch.reshape(r, (-1, 1))
            read_prob, site_ratio = model(S, X)
            loss1 = loss_fn_1(read_prob, y)
            loss2 = loss_fn_2(site_ratio, r)*10
            test_loss = loss1 + loss2
            # test_loss = loss1
            pred = read_prob > 0.5
            pred = pred.type(torch.int32)
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    if type == "val":
        print(f"Validate: \n Acc 0: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, r, site_ratio
    if type == "test":
        print(f"test: \n Acc 0: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_ResiDNN(args, dataset, model):

    train_dl, val_dl, test_dl = dataset

    # device
    device = (
        "cuda"
        if torch.cuda.is_available() and args.cuda
        else "cpu"
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.config['train']['learning_rate'],
                                 weight_decay=args.config['train']['weight_decay'])
    torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.25, last_epoch=-1)
    loss_fn_1 = torch.nn.BCELoss()
    loss_fn_2 = torch.nn.MSELoss()
    best = 10000

    for epoch in range(args.config['train']['epoch']):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dl, model, loss_fn_1, loss_fn_2, optimizer, device)
        test_loss, r, site_ratio = evaluate(val_dl, model, loss_fn_1, loss_fn_2, device)
        evaluate(test_dl, model, loss_fn_1, loss_fn_2, device,  type="test")

        if test_loss < best:
            best_epoch = epoch
            best = test_loss
            patience = 0

        else:
            patience += 1

        if patience == args.config['train']['early_stopping']:
            print("Ratio: %s; \n Predict: %s" % (r.cpu().numpy(), site_ratio.cpu().numpy()))
            break

    # save model
    PATH = args.out + "/" + args.motif + ".mod"
    torch.save(model, PATH)
