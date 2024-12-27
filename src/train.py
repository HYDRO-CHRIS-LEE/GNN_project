from src.EarlyStopping import *
from tqdm import tqdm
import torch

def train_astgcn(static_edge_index, num_epochs, model, train_loader, optimizer, loss_fn, store_path):
    early_stopper = EarlyStopping(store_path, patience=50, min_delta=0.0001)

    # Initialize a list to store average losses for each epoch
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        count_batches = 0

        for encoder_inputs, labels in tqdm(train_loader):
            y_hat = model(encoder_inputs, static_edge_index)
            loss = loss_fn(y_hat, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            count_batches += 1

        avg_loss = total_loss / count_batches
        epoch_losses.append(avg_loss)  # Store the average loss for this epoch
        print(f"Epoch {epoch+1} train Loss: {avg_loss:.4f}")

        if early_stopper.should_stop(model, avg_loss):
            print("Early stopping triggered")
            break

        torch.save(model, store_path)

def train_gwnet(num_epochs, model, device, train_loader, optimizer, loss_fn, supports, store_path):
    early_stopper = EarlyStopping(store_path, patience=50, min_delta=0.0001)

    # Initialize a list to store average losses for each epoch
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        count_batches = 0

        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, supports)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count_batches += 1

        avg_loss = total_loss / count_batches
        epoch_losses.append(avg_loss)  # Store the average loss for this epoch
        print(f"Epoch {epoch+1} train Loss: {avg_loss:.4f}")

        if early_stopper.should_stop(model, avg_loss):
            print("Early stopping triggered")
            break

        torch.save(model, store_path)

def train_tgcn(num_epochs, model, train_loader, optimizer, loader, loss_fn, store_path, device):
    # Ensure edge_index is a torch.LongTensor
    edge_index = torch.tensor(loader.edges, dtype=torch.long).to(device)
    edge_weight = torch.tensor(loader.edge_weights, dtype=torch.float).to(device)

    early_stopper = EarlyStopping(store_path, patience=50, min_delta=0.0001)
    epoch_losses = []
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        count_batches = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs[:, :, :, -1], edge_index, edge_weight=edge_weight)
                loss = loss_fn(outputs, targets)  # Predict the next time step
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                count_batches += 1

                pbar.set_postfix(loss=train_loss / (pbar.n + 1))
                pbar.update(1)

        avg_loss = train_loss / count_batches
        epoch_losses.append(avg_loss)  # Store the average loss for this epoch
        print(f"Epoch {epoch+1} train Loss: {avg_loss:.4f}")

        if early_stopper.should_stop(model, avg_loss):
            print("Early stopping triggered")
            break

    torch.save(model, store_path)