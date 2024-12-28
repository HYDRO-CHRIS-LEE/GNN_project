import argparse
import torch
import numpy as np
from torch_geometric_temporal.signal import temporal_signal_split

from SoyangDataLoader import *

from ASTGCN import *
from GraphWaveNet import *
from TGCN import *

from src.loss import *
from src.train import *
from src.eval import *

from visualization import *

# Prepare DataLoader
def create_dataloader(temporal_dataset, batch_size, shuffle, device):
    inputs = torch.from_numpy(np.array(temporal_dataset.features)).type(torch.FloatTensor).to(device)  # (B, N, F, T)
    targets = torch.from_numpy(np.array(temporal_dataset.targets)).type(torch.FloatTensor).to(device)  # (B, N, T)
    tensor_dataset = torch.utils.data.TensorDataset(inputs, targets)
    return torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def create_dataloader_for_gwnet(temporal_dataset, batch_size, shuffle, device):
    inputs = torch.from_numpy(np.array(temporal_dataset.features)).type(torch.FloatTensor).permute(0, 2, 1, 3).to(device)  # (B, N, F, T) -> (B, F, N, T)
    targets = torch.from_numpy(np.array(temporal_dataset.targets)).type(torch.FloatTensor).permute(0, 2, 1).to(device)  # (B, N, T) -> (B, T, N)
    tensor_dataset = torch.utils.data.TensorDataset(inputs, targets)
    return torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def data_load(input_seq_len, output_seq_len, adj_method, ratio, batch_size, device, model_name, shuffle=False):
    if adj_method == "Euc":
        print("Euclidian adj!")
        ### Euc Loader
        loader = SoYangRiverDatasetLoader(adj="Euc")
    elif adj_method == "Euc_down":
        print("Euclidian downstream adj!")
        ### Euc_down Loader
        loader = SoYangRiverDatasetLoader(adj="Euc_down")
    elif adj_method == "Hyd":
        print("Hydrologic adj!")
        ### Hyd Loader
        loader = SoYangRiverDatasetLoader(adj="Hyd")
    elif adj_method == "Hyd_down":
        print("Hydrologic downstream adj!")
        ### Hyd_down Loader
        loader = SoYangRiverDatasetLoader(adj="Hyd_down")

    dataset = loader.get_dataset(num_timesteps_in=input_seq_len, num_timesteps_out=output_seq_len)
    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ",  dataset.snapshot_count)
    print(next(iter(dataset))) # Show first sample

    # Train-Test Split
    print("Splitting Train / Test...")
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=ratio)

    # Make Train, Test Loader
    if (model_name == "GWNET"):
        train_loader = create_dataloader_for_gwnet(train_dataset, batch_size, shuffle, device=device)
        test_loader = create_dataloader_for_gwnet(test_dataset, batch_size, shuffle, device=device)
    else:
        train_loader = create_dataloader(train_dataset, batch_size, shuffle, device=device)
        test_loader = create_dataloader(test_dataset, batch_size, shuffle, device=device)

    # supports variable for Graph WaveNet Model
    supports = [torch.tensor(loader.A, dtype=torch.float).to(device)]

    return loader, train_dataset, test_dataset, train_loader, test_loader, supports


def create_model(in_time, out_time, station_num, feature_num, time_strides, model_name, dropout, blocks, layers, supports, learning_rate, DEVICE):
    # Create model
    if (model_name == "ASTGCN"):
        model = ASTGCN(nb_block=3, in_channels=feature_num, K=5, nb_chev_filter=5, nb_time_filter=3, time_strides=time_strides, num_for_predict=out_time, len_input=in_time, num_of_vertices=station_num, normalization="sym").to(DEVICE)
    elif (model_name == "GWNET"):
        model = gwnet(device=DEVICE, num_nodes=station_num, in_dim=feature_num, out_dim=out_time, dropout=dropout, supports=supports, blocks=blocks, layers=layers).to(DEVICE)
    elif (model_name == "TGCN"):
        model = TGCN(in_channels = feature_num, out_channels = out_time).to(DEVICE)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    print('Net\'s state_dict...')
    total_param = 0
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)
    #--------------------------------------------------
    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    return model, optimizer, loss_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adj", type=str, help="Select your adjaceny matrix from Hyd, Hyd_down, Euc, and Euc_down")
    parser.add_argument("model", type=str, help="Select your model from ASTGCN, GWNET, and TGCN")
    parser.add_argument("in_time", type=int, help="Input time for model", action="store")
    parser.add_argument("out_time", type=int, help="Output time for model", action="store")
    parser.add_argument("--time_strides", type=int, default=6, help="Input your time strides for your model. The input time should be divided by time strides.")
    parser.add_argument("--lr", type=float, default=8e-5, help="Input your appropriate learning rate for your model")
    parser.add_argument("--cuda", type=str, default='cuda:0', help="Select your cuda. Input format; cuda:0, cuda:1 ...")
    parser.add_argument("--station_num", type=int, default=22, help="Input your station numbers")
    parser.add_argument("--feature_num", type=int, default=4, help="Input your feature number")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train and Test ratio", action="store")
    parser.add_argument("--num_epochs", type=int, default=2000, help="Number of epochs", action="store")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of batch size", action="store")
    parser.add_argument("--only_eval", type=bool, default=False, help="If you want to do only evaluation")
    parser.add_argument("--dropout", type=float, default=0.1, help="Input your dropout rate for GraphWaveNet model")
    parser.add_argument("--blocks", type=int, default=4, help="Input your block numbers for GraphWaveNet model")
    parser.add_argument("--layers", type=int, default=2, help="Input your layer numbers for GraphWaveNet model")
    args = parser.parse_args()

    model_name = args.model            # Selected model for training
    batch_size = args.batch_size       # batch size for training
    input_seq_len = args.in_time       # Input seq lenth for your dataset
    output_seq_len = args.out_time     # Output seq lenth for your dataset
    time_strides = args.time_strides   # time_strides
    learning_rate = args.lr            # learning rate for training
    station_num = args.station_num     # Number of stations in your dataset
    feature_num = args.feature_num     # Number of features in your dataset
    ratio = args.ratio                 # Train-Test split ratio
    num_epochs = args.num_epochs       # Input your number of epochs
    adj_method = args.adj              # Pick your adjacency matrix
    only_eval = args.only_eval         # If you want to do only eval
    cuda = args.cuda                   # Your cuda number
    dropout = args.dropout             # Your dropout rate
    blocks = args.blocks
    layers = args.layers

    
    if model_name == "GWNET":
        figsave_casename = f"Model_{model_name}_Input_{input_seq_len}_Adj_{adj_method}_blocks{blocks}_layers{layers}"
        store_path = f'/home/data/figure_hydroml/model/{model_name}_{figsave_casename}.pt'
    else:
        figsave_casename = f"Model_{model_name}_Input_{input_seq_len}_Adj_{adj_method}"
        store_path = f'/home/data/figure_hydroml/model/{model_name}_{figsave_casename}.pt'

    figsave_path = "/home/data/figure_hydroml/"
    
    print("Checking your GPU env...")
    DEVICE = torch.device(cuda if torch.cuda.is_available() else 'cpu')

    print("Your Device is...", DEVICE)
    print("Your PyTroch Version is...", torch.__version__)
    print("Your batch size is...", batch_size)

    print("#############################")
    print("Load Soyang River Dataset...")

    loader, train_dataset, test_dataset, train_loader, test_loader, supports = data_load(input_seq_len, output_seq_len, adj_method, ratio, batch_size, DEVICE, model_name)
    model, optimizer, loss_fn = create_model(input_seq_len, output_seq_len, station_num, feature_num, time_strides, model_name, dropout, blocks, layers, supports, learning_rate, DEVICE)

    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(DEVICE)
        break;
    
    print("#############################")
    print("Your Model is...", model_name)

    # print("Training start...")
    # if (model_name == "ASTGCN"):
    #     train_astgcn(static_edge_index, num_epochs, model, train_loader, optimizer, loss_fn, store_path)
    # elif (model_name == "GWNET"):
    #     train_gwnet(num_epochs, model, DEVICE, train_loader, optimizer, loss_fn, supports, store_path)
    # elif (model_name == "TGCN"):
    #     train_tgcn(num_epochs, model, train_loader, optimizer, loader, loss_fn, store_path, DEVICE)
    # print("Training complete!")

    # print("Evaluation start...")
    # if (model_name == "ASTGCN"):
    #     predictions, test_labels = eval_astgcn(model, test_loader, static_edge_index, loss_fn, only_eval, store_path, DEVICE)
    # elif (model_name == "GWNET"):
    #     predictions, test_labels = eval_gwnet(model, test_loader, loss_fn, supports, only_eval, store_path, DEVICE)
    # elif(model_name == "TGCN"):
    #     predictions, test_labels = eval_tgcn(model, test_loader, loss_fn, loader, only_eval, store_path, DEVICE)
    # print("Evaluation done")

    if not only_eval:
        print("Training start...")
        if (model_name == "ASTGCN"):
            train_astgcn(static_edge_index, num_epochs, model, train_loader, optimizer, loss_fn, store_path)
        elif (model_name == "GWNET"):
            epoch_losses, adaptive_adj_matrices = train_gwnet(
                num_epochs,
                model,
                DEVICE,
                train_loader,
                optimizer,
                loss_fn,
                supports,
                store_path
            )
            # If desired, save adjacency to a file
            np.save(f"adap_adj_Model_{model_name}_Input_{input_seq_len}_Adj_{adj_method}_blocks{blocks}_layers{layers}.npy", adaptive_adj_matrices)
            print("Adaptive adjacency matrices saved to disk.")

        elif (model_name == "TGCN"):
            train_tgcn(num_epochs, model, train_loader, optimizer, loader, loss_fn, store_path, DEVICE)
        print("Training complete!")
    else:
        print("Skipping training as only_eval is set to True")

    print("Evaluation start...")
    if (model_name == "ASTGCN"):
        predictions, test_labels = eval_astgcn(
            model=model,
            loader=loader,  
            test_loader=test_loader,
            static_edge_index = static_edge_index,
            adjmethod=args.adj,
            loss_fn=loss_fn,
            only_eval=args.only_eval,
            store_path=store_path,
            DEVICE=DEVICE,
            input_seq_len=input_seq_len,  # Pass input_seq_len from args
        )
        #predictions, test_labels = eval_astgcn(model, test_loader, static_edge_index, loss_fn, only_eval, store_path, DEVICE)
    elif (model_name == "GWNET"):
        predictions, test_labels = eval_gwnet(
            model=model,
            loader=loader,
            test_loader=test_loader,
            adjmethod=args.adj,
            loss_fn=loss_fn,
            supports=supports,
            only_eval=args.only_eval,
            store_path=store_path,
            DEVICE=DEVICE,
            input_seq_len=input_seq_len,  # Pass input_seq_len from args
            blocks=args.blocks,           # Pass blocks for GWNET
            layers=args.layers            # Pass layers for GWNET
        )

        #predictions, test_labels = eval_gwnet(model, test_loader, loss_fn, supports, only_eval, store_path, DEVICE)
    elif(model_name == "TGCN"):
        predictions, test_labels = eval_tgcn(
            model=model,
            loader=loader, 
            test_loader=test_loader,
            adjmethod=args.adj,
            loss_fn=loss_fn,
            only_eval=args.only_eval,
            store_path=store_path,
            DEVICE=DEVICE,
            input_seq_len=input_seq_len,  # Pass input_seq_len from args
        )
        #predictions, test_labels = eval_tgcn(model, test_loader, loss_fn, loader, only_eval, store_path, DEVICE)
    print("Evaluation done")

    print("Visualization start...")
    # vi = visualization(loader, predictions, test_labels, figsave_path, figsave_casename, loss_fn)
    # vi.scattor_plot()
    # vi.time_series()
    # vi.metrics()
    # vi.plot_bar()

if __name__ == "__main__":
    main()