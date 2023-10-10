import torch


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (S, X, y) in enumerate(dataloader):
        S, X, y = S.to(device), X.to(device), y.to(device)
        y = torch.reshape(y, (-1, 1))

        # Compute prediction error
        site_predict, read_predict_site = model(S, X)
        loss1 = loss_fn(site_predict, y)
        loss2 = loss_fn(read_predict_site, y)
        loss = loss1 + loss2

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def evaluate(dataloader, model, loss_fn, device, type="val"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct1, correct2, = 0, 0, 0
    with torch.no_grad():
        for  S, X, y in dataloader:
            S, X, y = S.to(device), X.to(device), y.to(device)
            y = torch.reshape(y, (-1, 1))
            site_predict, read_predict_site = model(S, X)
            loss1 = loss_fn(site_predict, y)
            loss2 = loss_fn(read_predict_site, y)
            test_loss = loss1 + loss2
            pred = site_predict > 0.5
            pred = pred.type(torch.int32)
            correct1 += (pred == y).type(torch.float).sum().item()
            pred = read_predict_site > 0.5
            pred = pred.type(torch.int32)
            correct2 += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct1 /= size
    correct2 /= size
    if type == "val":
        print(f"Validate: \n Acc 0: {(100*correct1):>0.1f}%, Acc 1: {(100*correct2):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if type == "test":
        print(f"test: \n Acc 0: {(100*correct1):>0.1f}%, Acc 1: {(100*correct2):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_MutliDNN(args, dataset, model):

    train_dl, val_dl, test_dl = dataset

    # device
    device = (
        "cuda"
        if torch.cuda.is_available() and args.cuda
        else "cpu"
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.config['train']['learning_rate'])
    loss_fn = torch.nn.BCELoss()

    for epoch in range(args.config['train']['epoch']):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dl, model, loss_fn, optimizer, device)
        evaluate(val_dl, model, loss_fn, device)
        evaluate(test_dl, model, loss_fn, device,  type="test")

    # save model
    PATH = args.out + "/" + args.motif + ".mod"
    torch.save(model, PATH)
