import torch
from tqdm import tqdm
from save_predictions import save_model_predictions

def eval_astgcn(model, loader, test_loader, adjmethod, static_edge_index, loss_fn, only_eval, store_path, DEVICE, input_seq_len=None):
    # if only_eval:
    #     model = torch.load(store_path, map_location=DEVICE)
    if only_eval:
        loaded_model = torch.load(store_path, map_location=DEVICE)
        if isinstance(loaded_model, dict):
            model.load_state_dict(loaded_model)
        else:
            model = loaded_model


    model.eval()
    # Store for analysis
    total_loss = []
    test_labels = []
    predictions = []
    
    for encoder_inputs, labels in tqdm(test_loader):
        # Get model predictions
        y_hat = model(encoder_inputs, static_edge_index)
        # Mean squared error
        loss = loss_fn(y_hat, labels)
        total_loss.append(loss.item())
        # Store for analysis below
        # test_labels.append(labels)
        # predictions.append(y_hat)

        test_labels.append(labels.cpu().detach())  # Move to CPU
        predictions.append(y_hat.cpu().detach())  # Move to CPU
    
    print("Test Loss: {:.4f}".format(sum(total_loss)/len(total_loss)))

    # Concatenate along the first dimension (time) and remove any singleton dimensions
    predictions_tensor = torch.cat(predictions, dim=0).squeeze()  # Remove all singleton dimensions
    test_labels_tensor = torch.cat(test_labels, dim=0).squeeze()  # Remove all singleton dimensions

    print(f"Predictions shape after squeeze: {predictions_tensor.shape}")
    print(f"Test Labels shape after squeeze: {test_labels_tensor.shape}")

    save_model_predictions(
        predictions_tensor.numpy(),
        test_labels_tensor.numpy(),
        loader.data_center_idx_mapping,
        adjmethod,
        '/home/data/figure_hydroml/',  # Update this path
        'ASTGCN',
        input_seq_len=input_seq_len
    )

    return predictions, test_labels

def eval_gwnet(model, loader, test_loader, adjmethod, loss_fn, supports, only_eval, store_path, DEVICE, input_seq_len=None, blocks=None, layers=None):
    if only_eval:
        loaded_model = torch.load(store_path, map_location=DEVICE)
        if isinstance(loaded_model, dict):
            model.load_state_dict(loaded_model)
        else:
            model = loaded_model

    model.eval()
    total_loss = []
    test_labels = []
    predictions = []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            y_hat = model(data, supports)
            total_loss.append(loss_fn(y_hat, target).item())
            
            # # 타겟 차원 변경 (기존과 동일)
            # target = target.permute(0, 2, 1)
            # test_labels.append(target)
            
            # # y_hat에 대한 permute를 차원에 맞게 수정
            # y_hat = y_hat.permute(0, 2, 1)
            # predictions.append(y_hat)
            target = target.permute(0, 2, 1)
            test_labels.append(target.cpu().detach())  # Move to CPU
            y_hat = y_hat.permute(0, 2, 1)
            predictions.append(y_hat.cpu().detach())  # Move to CPU

    print("Test Loss: {:.4f}".format(sum(total_loss)/len(total_loss)))

    # Concatenate along the first dimension (time) and remove any singleton dimensions
    predictions_tensor = torch.cat(predictions, dim=0).squeeze()  # Remove all singleton dimensions
    test_labels_tensor = torch.cat(test_labels, dim=0).squeeze()  # Remove all singleton dimensions

    print(f"Predictions shape after squeeze: {predictions_tensor.shape}")
    print(f"Test Labels shape after squeeze: {test_labels_tensor.shape}")

    save_model_predictions(
        predictions_tensor.numpy(),
        test_labels_tensor.numpy(),
        loader.data_center_idx_mapping,
        adjmethod,
        '/home/data/figure_hydroml/',  # Update this path
        'GWNET',
        input_seq_len=input_seq_len,  # Pass input_seq_len
        blocks=blocks,                # Pass blocks
        layers=layers                 # Pass layers
    )

    return predictions, test_labels


def eval_tgcn(model, loader, test_loader, adjmethod, loss_fn, only_eval, store_path, DEVICE, input_seq_len=None):
    # if only_eval:
    #     model = torch.load(store_path, map_location=DEVICE)
    if only_eval:
        loaded_model = torch.load(store_path, map_location=DEVICE)
        if isinstance(loaded_model, dict):
            model.load_state_dict(loaded_model)
        else:
            model = loaded_model

    edge_index = torch.tensor(loader.edges, dtype=torch.long).to(DEVICE)
    edge_weight = torch.tensor(loader.edge_weights, dtype=torch.float).to(DEVICE)

    # Testing loop
    model.eval()
    total_loss = []
    test_labels = []
    predictions = []
    for data, target in tqdm(test_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        y_hat = model(data[:, :, :, -1], edge_index, edge_weight=edge_weight)
        total_loss.append(loss_fn(y_hat, target).item())
        # test_labels.append(target)
        # predictions.append(y_hat)
        test_labels.append(target.cpu().detach())  # Move to CPU
        predictions.append(y_hat.cpu().detach())  # Move to CPU

    print("Test Loss: {:.4f}".format(sum(total_loss)/len(total_loss)))

    # Concatenate along the first dimension (time) and remove any singleton dimensions
    predictions_tensor = torch.cat(predictions, dim=0).squeeze()  # Remove all singleton dimensions
    test_labels_tensor = torch.cat(test_labels, dim=0).squeeze()  # Remove all singleton dimensions

    save_model_predictions(
        predictions_tensor.numpy(),
        test_labels_tensor.numpy(),
        loader.data_center_idx_mapping,
        adjmethod,
        '/home/data/figure_hydroml/',  # Update this path
        'ASTGCN',
        input_seq_len=input_seq_len
    )

    return predictions, test_labels