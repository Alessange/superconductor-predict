import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
import torch
from cgcnn.data import CIFData
from torch.utils.data import DataLoader
from SupTran import Transformer
from trans_embedding import CrystalGraphConvNet
import numpy as np

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "/home/sxhuang/superconductor/cgcnn_embedding/cgcnn-master/data/dataset"
test_path = "/home/sxhuang/superconductor/cgcnn_embedding/cgcnn-master/data/sample-regression"
# Load the dataset
dataset = CIFData(path)
structures, _, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]
cgcnn = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len)
cgcnn.eval()

# Set collate_fn
def collate_fn(data_list):
    """
    :data_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int
    :return: tuple of embedding features tensor
    """
    concate_input = []
    target_input = []
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) \
            in enumerate(data_list):
        n_i = atom_fea.shape[0]
        crystal_atom_idx = [torch.LongTensor(np.arange(n_i))]
        embedding_tensor = cgcnn(atom_fea, nbr_fea, nbr_fea_idx,
                                 crystal_atom_idx)
        concate_input.append(embedding_tensor.unsqueeze(0))
        target_input.append(target.unsqueeze(0))
    return torch.cat(concate_input, dim=0).transpose(0,1), torch.cat(target_input, dim=0).unsqueeze(0)


dataloader = DataLoader(dataset, batch_size=256, shuffle=True,
                        collate_fn=collate_fn)
# Model parameter
model_param = {'embedding_size': 128, 'num_head': 8, 'num_hid': 8,
               'num_layer': 8, 'dropout': 0.01}
# Load model
model = Transformer(d_model=model_param['embedding_size'],
                    nhead=model_param['num_head'], nhid=model_param['num_hid'],
                    nlayers=model_param['num_layer'],
                    dropout=model_param['dropout'], device=device).to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = MSELoss()


# Define the training function
def train(dataloader_, model_, criterion_, optimizer_, device_):
    model.train()

    for epoch in range(100):  # 100 epochs, you can change this
        print(epoch)
        epoch_loss = 0
        epoch_mae = 0
        for batch_idx, (data, target) in enumerate(dataloader_):
            data, target = data.to(device_), target.to(device_)
            optimizer.zero_grad()

            output = model_(data)
            loss = criterion_(output, target)

            mae = torch.mean(torch.abs(output - target))
            loss.backward()
            optimizer_.step()
            epoch_loss += loss.item()
            epoch_mae += mae.item()
            scheduler.step()
        # Save the model
        # torch.save(model.state_dict(), f"/home/sxhuang/superconductor/model/pre_train_model/epoch_one/model_epoch_{epoch}.pth")
        # Print the loss and accuracy for each epoch
        print(f"Epoch {epoch}: Loss = {epoch_loss / len(dataloader)}, MAE = {epoch_mae / len(dataloader)}")
        # Save the loss and accuracy into a file
        with open("/home/sxhuang/superconductor/cgcnn_embedding/cgcnn-master/data/data_ana/train1.txt",'a') as f:
            f.write(f"Epoch {epoch}: Loss = {epoch_loss / len(dataloader)}, MAE = {epoch_mae / len(dataloader)}\n")

        # Save the model
    torch.save(model.state_dict(),
               f"/home/sxhuang/superconductor/cgcnn_embedding/cgcnn-master/model/super_model.pt")


# Train the model
if __name__ == '__main__':
    train(dataloader, model, criterion, optimizer, device)
