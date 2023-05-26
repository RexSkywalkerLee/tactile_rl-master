import argparse
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import os

from sklearn.decomposition import PCA
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj, save_obj
import datetime
import pickle

def time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

# set random seed
torch.manual_seed(0)
np.random.seed(0)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--regress_num", type=int, default=10) #5
parser.add_argument("--frame_num", type=int, default=200)
parser.add_argument("--test_on_unseen", action="store_true")
parser.add_argument("--output_dir", type=str, default="./data/output/regression")
args = parser.parse_args()

output_dir = args.output_dir + time_str()
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)


class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(85, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(16 * args.frame_num, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp2(x)
        return x


class MLPNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(85, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 126),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, args.regress_num),
        )

    def forward(self, x):
        #x = x.reshape(x.size(0), -1, self.stack_num * 85)
        x = self.mlp1(x)
        x = x.mean(dim=1)
        y1 = self.mlp2(x)
        y2 = self.mlp3(x)
        return y1, y2


class TemporalConv(nn.Module):
    def __init__(self):
        super().__init__()
        print("Instantiated Temporal CNN")
        self.mlp1 = nn.Sequential(
            nn.Conv1d(85, 64, 5, 2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, 2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(128, 256, 5, 2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(256, 256, 5, 2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(256, 256, 5, 2, padding=1)
        )

        self.mlp2 = nn.Sequential(nn.Linear(256 * 5, 512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(512, 126))

        self.mlp3 = nn.Sequential(nn.Linear(256 * 5, 512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(512, args.regress_num))

    def forward(self, x):
        #x = x.reshape(x.size(0), -1, self.stack_num * 85)
        x = x.permute(0, 2, 1)
        x = self.mlp1(x)
        x = x.reshape(x.size(0), -1)
        y1 = self.mlp2(x)
        y2 = self.mlp3(x)
        return y1, y2
# x = x.mean(dim=1)
# y1 = self.mlp2(x)
# y2 = self.mlp3(x)
# return y1, y2

def get_dir_files(path, postfix, return_items=False):
    '''
    Get files of certain postfix in a directory.
    :param path: the directory to enumerate, absolute path.
    :param postfix: postfix of filename.
    :return: a list of absolute paths.
    '''

    filelist = []
    items = []
    
    files = os.listdir(path)
    for item in files:
        if os.path.isfile(os.path.join(path, item)):
            if item.endswith(postfix):
                filelist.append(os.path.join(path, item))
                items.append(item)
        # if len(filelist) > 1:
        #     break
    if return_items:
        return filelist, items
    
    return filelist

def load_all_data(path):
    data = []
    all_trajs = get_dir_files(path, '.npy')
    for traj in all_trajs:
        chunk = np.load(traj, allow_pickle=True)
        data.append(chunk)
        print(chunk[0].keys())
    data = np.concatenate(data)
    print(data.shape)
    return data

def load_all_real_data(path):
    data = []
    all_real_trajs, items = get_dir_files(path, '.pkl', True)
    for traj, item in zip(all_real_trajs, items):
        chunk = np.load(traj, allow_pickle=True)
        print(item)
        data.append([item[:-4], chunk['obs'][:200, 0, :85]])
        print(chunk['obs'][:200, 0, :85].shape)
    return data

def read_index_mapping(file):
    import json
    with open(file) as f:
        data = json.load(f)['list']
    print(data)
    return data

if __name__ == "__main__":
    # load_real_data
    real_data = load_all_real_data("../real_data/mar07_23/recon")
    
    # load data
    key_order = read_index_mapping("../polygon_data/polygon_asset.txt")
    gt_offsets = np.load("../polygon_data/output/deform_verts.npz", allow_pickle=True)
    gt_keys = [key[:-4] for key in gt_offsets]
    #print(gt_keys)
    remapping_indexes = [gt_keys.index(key) for key in key_order] # y_label to mesh label in all_feats

    data = load_all_data("/code/PolygonTrajectoryTrained")# np.load("/code/PolygonTrajectory", allow_pickle=True)
    gt_offsets = {i: gt_offsets[filename] for i, filename in enumerate(gt_offsets.files)} #
    #print(gt_offsets)

    all_feats = np.stack([gt_offsets[key] for key in gt_offsets], axis=0)
    all_feats = all_feats.reshape(all_feats.shape[0], -1)
    #print(all_feats.shape)
    pca = PCA(n_components=args.regress_num)
    pca.fit(all_feats)

    with open(os.path.join(output_dir, f"pca.pkl"), 'wb') as pickle_file:
        pickle.dump(pca, pickle_file)
        
    def save_offset_as_mesh(y, output_path):
        deform_verts_pca = y.clone().detach().cpu().numpy()
        deform_verts = pca.inverse_transform(deform_verts_pca.reshape(1, -1))[0].reshape(-1, 3)
        src_mesh = ico_sphere(4)
        new_src_mesh = src_mesh.offset_verts(torch.from_numpy(deform_verts).float())
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        save_obj(output_path, final_verts, final_faces)

    gt_offsets_pca = {key: pca.transform(gt_offsets[key].reshape(1, -1))[0] for key, value in gt_offsets.items()}
    gt_offsets = {key: torch.from_numpy(value).float().to(device) for key, value in gt_offsets_pca.items()}

    states = [seq["state"] for seq in data]
    clipped_states = np.stack([state[: args.frame_num] for state in states], axis=0)
    labels = np.asarray([remapping_indexes[seq["obj"][0, 0]] for seq in data])
    all_labels = list(set(list(labels)))

    import random
    random.shuffle(all_labels)
    split_point = int(len(all_labels) * 0.2)
    test_labels, train_labels = all_labels[:split_point], all_labels[split_point:]
    #print(test_labels, train_labels)
    # convert to torch
    clipped_states = torch.from_numpy(clipped_states).float().to(device)
    labels = torch.from_numpy(labels).long().to(device)

    if False:#not args.test_on_unseen:
        # tensor_dataset
        dataset = torch.utils.data.TensorDataset(clipped_states, labels)

        # split data
        train_size = int(0.8 * len(clipped_states))
        test_size = len(clipped_states) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    else:
        seen_classes = train_labels#[0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 15]
        # unseen_classes = [1, 3, 9, 11, 14]

        seen_idx = torch.tensor([label in seen_classes for label in labels])

        seen_states, seen_labels = clipped_states[seen_idx], labels[seen_idx]
        unseen_states, unseen_labels = clipped_states[~seen_idx], labels[~seen_idx]

        train_dataset = torch.utils.data.TensorDataset(seen_states, seen_labels)
        test_dataset = torch.utils.data.TensorDataset(unseen_states, unseen_labels)

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # model
    model = TemporalConv().to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    # loss
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    # train
    for epoch in range(args.epochs):
        model.train()

        epoch_cls_loss, epoch_reg_loss, epoch_loss = 0, 0, 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            reg_y = torch.stack([gt_offsets[label] for label in y.tolist()], dim=0)
            reg_y = reg_y.reshape(reg_y.shape[0], -1)

            cls_y_pred, reg_y_pred = model(x)

            # print(cls_y_pred.shape, reg_y_pred.shape, reg_y.shape, y.shape, y)
            reg_loss = reg_criterion(reg_y_pred, reg_y) * 10
            cls_loss = cls_criterion(cls_y_pred, y)

            loss = reg_loss + cls_loss

            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch}, Reg Loss: {epoch_reg_loss / len(train_loader):.4f}, "
                f"Cls Loss: {epoch_cls_loss / len(train_loader):.4f}, "
                f"Loss: {epoch_loss / len(train_loader):.4f}"
            )

        # test
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                reg_loss = 0
                total, correct = 0, 0
                for i, (x, y) in enumerate(test_loader):

                    reg_y = torch.stack([gt_offsets[label] for label in y.tolist()], dim=0)
                    reg_y = reg_y.reshape(reg_y.shape[0], -1)

                    #print(x.shape)
                    cls_y_pred, reg_y_pred = model(x)

                    reg_loss = reg_criterion(reg_y_pred, reg_y) * 10
                    reg_loss += reg_loss.item()

                    _, predicted = torch.max(cls_y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

                print(f"Test Reg Loss: {reg_loss / len(test_loader):.4f}, " f"Test Acc: {correct / total:.4f}")

                predicted_real_meshes = []
                for item, data in real_data:
                    x = torch.from_numpy(data).unsqueeze(0).cuda()
                    #print(x.shape)
                    _, reg_y_real = model(x)
                    predicted_real_meshes.append(reg_y_real[0])
                    
                if epoch % 10 == 0:
                    for i in range(int(y.size(0))):
                        save_offset_as_mesh(
                            reg_y_pred[i], os.path.join(output_dir, f"epoch_{epoch:06}_id_{y[i].item()}.obj")
                        )
                        
                        save_offset_as_mesh(
                            reg_y[i], os.path.join(output_dir, f"epoch_{epoch:06}_gt_id_{y[i].item()}.obj")
                        )
                    
                    for i, predicted_real_mesh in enumerate(predicted_real_meshes):
                        save_offset_as_mesh(
                            predicted_real_mesh, os.path.join(output_dir, f"epoch_{epoch:06}_pred_{real_data[i][0]}.obj")
                        )
                    
                    torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch:06}.pth"))
                    
