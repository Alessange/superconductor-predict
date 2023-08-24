from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
import torch
from Data Processing.dataset import TrainDataset
from Model.transformer import Transformer, PadPos, PadAtom

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    # Unzip the batch
    sequences, mask_targets, original_targets = zip(*batch)
    sequences = list(sequences)
    # Define pad elements
    pad_atom = PadAtom()
    pad_pos = PadPos()

    for i in range(len(sequences)):
        sentence = sequences[i]
        pad_num = 864 - len(sentence)
        if pad_num > 0:
            pad_sequence = [pad_atom, pad_pos] * (pad_num // 2)
            if pad_num % 2 != 0:
                pad_sequence.append(pad_atom)
            # Add padding to the sequence
            sequences[i] = sentence + pad_sequence
    sequences = tuple(sequences)
    # Stack targets into a tensor
    mask_targets = torch.stack(mask_targets).squeeze(1)
    original = torch.stack(original_targets).squeeze(1).transpose(0,1)
    return sequences, mask_targets, original


# Load the dataset
dataset = TrainDataset('/Users/huangshixun/Desktop/Transformer'
                       '/superconductors/data/pre_train_new.json')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                        collate_fn=collate_fn)
# Model parameter
model_param = {'embedding_size': 25, 'num_head': 25, 'num_hid': 25,
               'num_layer': 8, 'dropout': 0.1}
# Load model
model = Transformer(d_model=model_param['embedding_size'],
                    nhead=model_param['num_head'], nhid=model_param['num_hid'],
                    nlayers=model_param['num_layer'],
                    dropout=model_param['dropout'], device=device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = MSELoss()


# Define the training function
def train(dataloader_, model_, criterion_, optimizer_, device_):
    model.train()

    for epoch in range(100):  # 100 epochs, you can change this
        epoch_loss = 0
        epoch_mae = 0

        for batch_idx, (data, mask_target, original) in enumerate(dataloader_):
            data, mask_target, original = data, mask_target.to(
                device_), original.to(device_)

            optimizer.zero_grad()
            output = model_(data, mask_target)
            loss = criterion_(output, original)
            print(loss)
            mae = torch.mean(torch.abs(output - original))
            loss.backward()
            optimizer_.step()

            epoch_loss += loss.item()
            epoch_mae += mae.item()
        scheduler.step()

        # Save the model
        # torch.save(model.state_dict(), f"/home/sxhuang/superconductor/model/pre_train_model/epoch_one/model_epoch_{epoch}.pth")

        # Print the loss and accuracy for each epoch
        print(
            f"Epoch {epoch}: Loss = {epoch_loss / len(dataloader)}, MAE = {epoch_mae / len(dataloader)}")

        # Save the loss and accuracy into a file
        break
        """
        with open("/home/sxhuang/superconductor/data_analysis/train_log.txt", 'a') as f:
            f.write(
                f"Epoch {epoch}: Loss = {epoch_loss / len(dataloader)}, MAE = {epoch_mae / len(dataloader)}\n")

        # Save the model
    torch.save(model.state_dict(), f"/home/sxhuang/superconductor/model/model_pre_train.pt")"""


# Train the model
train(dataloader, model, criterion, optimizer, device)
