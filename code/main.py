import torch
import numpy as np
from DiffusionModel import GaussianDiffusion_ST, Model_all, ST_Diffusion
from Models import HEM
from torch.optim import AdamW
import argparse
from Dataset import get_dataloader
import time
from discreteDataset import FS_Dataset
import pickle
import os
import random
from Embed import D2C


def split_list(elements, train_ratio):
    random.shuffle(elements)  # Shuffle the list
    n = len(elements)
    train_end = int(n * train_ratio)

    train_data = elements[:train_end]
    test_data = elements[train_end:]

    trainloader = get_dataloader(train_data, opt.batch_size, D=opt.dim, shuffle=True)
    testloader = get_dataloader(test_data, len(test_data) if len(test_data) <= 1000 else 1000, D=opt.dim, shuffle=False)
    valloader = testloader

    return trainloader, valloader, testloader


def update_event_and_interval_times(original_event_tensor, original_interval_tensor, replacing_interval_tensor):
    assert original_event_tensor.shape == original_interval_tensor.shape == replacing_interval_tensor.shape, "All tensors must have the same shape"

    batch_size, max_seq_len = original_event_tensor.shape
    update_success_count = 0
    update_fail_count = 0

    # Step 1: Update the Original Interval Time Tensor
    updated_interval_tensor = torch.zeros_like(original_interval_tensor)
    for i in range(batch_size):
        for j in range(0, max_seq_len):  # Exclude the last interval
            if (j != 0 and j != max_seq_len - 1 and original_interval_tensor[i, j] != 0
                    and original_interval_tensor[i, j+1] != 0 and original_interval_tensor[i, j] != replacing_interval_tensor[i, j]):
                # Check if the replacing interval is within the specified range and is non-zero
                if 0 < replacing_interval_tensor[i, j] < (original_interval_tensor[i, j] + original_interval_tensor[i, j + 1]):
                    updated_interval_tensor[i, j] = replacing_interval_tensor[i, j]
                    update_success_count += 1
                else:
                    updated_interval_tensor[i, j] = original_interval_tensor[i, j]
                    update_fail_count += 1
            else:
                updated_interval_tensor[i, j] = original_interval_tensor[i, j]

    # Step 2: Update the Original Event Time Tensor using the updated Interval Tensor
    res = torch.zeros_like(original_event_tensor)
    for i in range(batch_size):
        for j in range(0, max_seq_len):
            if j != 0 and original_interval_tensor[i, j] != 0 and original_interval_tensor[i, j] != updated_interval_tensor[i, j]:
                res[i, j] = original_event_tensor[i, j - 1] + updated_interval_tensor[i, j]
            else:
                res[i, j] = original_event_tensor[i, j]

    return res, update_success_count, update_fail_count


def separate_and_pad_tensor(input_tensor, lengths, max_length, first):
    batch_size = len(lengths)
    # Initialize the output tensor with zeros
    output_tensor = torch.zeros(batch_size, max_length)

    # Copy sequences into the output tensor
    start_idx = 0
    for i, length in enumerate(lengths):
        end_idx = start_idx + length
        output_tensor[i, 1:length+1] = input_tensor[start_idx:end_idx].view(-1)
        output_tensor[i, 0] = first
        start_idx = end_idx

    return output_tensor


def selective_replace_in_tuples(original_tuple, replacing_tuple, replace_rate):
    assert len(original_tuple) == len(replacing_tuple), "Tuples must be of the same length"
    for orig, repl in zip(original_tuple, replacing_tuple):
        assert orig.shape == repl.shape, "All corresponding tensors must be of the same shape"

    batch_size, max_seq_len = original_tuple[0].shape
    for i in range(batch_size):
        # Identify meaningful values (non-zero) excluding the first element for the first tensor
        meaningful_indices = torch.nonzero(original_tuple[0][i, 1:], as_tuple=True)[0] + 1

        # Calculate the number of elements to replace in this item
        num_replace = int(len(meaningful_indices) * replace_rate)

        # Randomly select indices to replace
        replace_indices = torch.randperm(len(meaningful_indices))[:num_replace]

        # Perform the replacement for each tensor in the tuple
        for orig, repl in zip(original_tuple, replacing_tuple):
            orig[i, meaningful_indices[replace_indices]] = repl[i, meaningful_indices[replace_indices]]


def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_name():
    TIME = int(time.time())
    TIME = time.localtime(TIME)
    return time.strftime("%Y-%m-%d %H:%M:%S", TIME)


def normalization(x, MAX, MIN):
    return (x - MIN) / (MAX - MIN)


def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--total_epochs', type=int, default=2000, help='')
    parser.add_argument('--machine', type=str, default='none', help='')
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2', 'Euclid'], help='')
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='')

    parser.add_argument('--dim', type=int, default=1, help='', choices=[1, 2, 3])
    parser.add_argument('--loc_emb_dim', type=int, default=32, help='')

    parser.add_argument('--dataset', type=str, default='FourSquareNYC',
                        choices=['FourSquareNYC', 'FourSquareTKY'], help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--timesteps', type=int, default=500, help='')
    parser.add_argument('--samplingsteps', type=int, default=500, help='')
    parser.add_argument('--objective', type=str, default='pred_noise', help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    # parser.add_argument('--seq_len', type=int, default=5, help='')
    # parser.add_argument('--generate_num', type=int, default=128, help='')
    parser.add_argument('--seq_len', type=int, default=40, help='')
    parser.add_argument('--generate_num', type=int, default=2000, help='')
    parser.add_argument('--training_name', type=int, default=10000, help='')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args


opt = get_args()
if opt.cuda_id == 'cpu':
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(opt.cuda_id))
    # device = torch.device("cuda")
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)


def data_loader():
    f = open('dataset/{}/data_train.pkl'.format(opt.dataset), 'rb')
    train_data = pickle.load(f)
    train_data = [[list(i) for i in u] for u in train_data]
    train_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u) if i[0] - u[index - 1][0] != 0] for u in
                  train_data]  # (User_len, Seq_len, 4)

    f = open('dataset/{}/data_val.pkl'.format(opt.dataset), 'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]
    val_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u) if i[0] - u[index - 1][0] != 0] for u in
                val_data]

    f = open('dataset/{}/data_test.pkl'.format(opt.dataset), 'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]
    test_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u) if i[0] - u[index - 1][0] != 0] for u in
                 test_data]

    data_all = train_data + test_data + val_data

    Max, Min = [], []
    for m in range(opt.dim + 2):
        if m > 0:
            Max.append(max([i[m] for u in data_all for i in u]))
            Min.append(min([i[m] for u in data_all for i in u]))
        else:
            Max.append(1)
            Min.append(0)

    assert Min[1] > 0


    train_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in train_data]
    test_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_data]
    val_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in val_data]

    # Add the first point
    first_checkin = [1e-10, 1e-10, 0.5, 0.5]
    train_data = [[first_checkin] + u for u in train_data]
    test_data = [[first_checkin] + u for u in test_data]
    val_data = [[first_checkin] + u for u in val_data]

    trainloader = get_dataloader(train_data, opt.batch_size, D=opt.dim, shuffle=True)
    testloader = get_dataloader(test_data, len(test_data) if len(test_data) <= 1000 else 1000, D=opt.dim, shuffle=False)
    valloader = get_dataloader(test_data, len(val_data) if len(val_data) <= 1000 else 1000, D=opt.dim, shuffle=False)

    return trainloader, testloader, valloader, Max, Min, first_checkin


def Batch2toModel(batch, transformer):
    if opt.dim == 1:
        event_time_origin, event_time, lng = map(lambda x: x.to(device), batch)
        event_loc = lng.unsqueeze(dim=2)

    if opt.dim == 2:
        event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)  # (B, Seq_len)

        event_loc = torch.cat((lng.unsqueeze(dim=2), lat.unsqueeze(dim=2)), dim=-1)  # (B, Seq_len, 2)

    if opt.dim == 3:
        event_time_origin, event_time, lng, lat, height = map(lambda x: x.to(device), batch)

        event_loc = torch.cat((lng.unsqueeze(dim=2), lat.unsqueeze(dim=2), height.unsqueeze(dim=2)), dim=-1)

    event_time = event_time.to(device)  # (B, Seq_len)
    event_time_origin = event_time_origin.to(device)  # (B, Seq_len)
    event_loc = event_loc.to(device)  # (B, Seq_len, 2)

    enc_out, mask = transformer(event_loc, event_time_origin)  # (B, Seq_len, dim), (B, Seq_len, 1)

    enc_out_non_mask = []
    event_time_non_mask = []
    event_loc_non_mask = []
    lengths = []
    for index in range(mask.shape[0]):
        length = int(sum(mask[index]).item())
        lengths.append(length - 1)
        if length > 1:
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length - 1]]
            event_time_non_mask += [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
            event_loc_non_mask += [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]

    assert sum(lengths) == len(enc_out_non_mask)
    enc_out_non_mask = torch.cat(enc_out_non_mask, dim=0)  # (B*Seq_len-1, dim)
    event_time_non_mask = torch.cat(event_time_non_mask, dim=0)  # (B*Seq_len-1)
    event_loc_non_mask = torch.cat(event_loc_non_mask, dim=0)  # (B*Seq_len-1)

    event_time_non_mask = event_time_non_mask.reshape(-1, 1, 1)  # (B*Seq_len-1, 1, 1)
    event_loc_non_mask = event_loc_non_mask.reshape(-1, 1, opt.dim)  # (B*Seq_len-1, 1, 2)

    enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0], 1, -1)  # (B*Seq_len-1, 1, dim)

    return event_time_non_mask, event_loc_non_mask, enc_out_non_mask, lengths


def DisBatch2toModel(batch, transformer, d2c):
    event_time_origin, event_time, locs = map(lambda x: x.to(device), batch)
    # event_locid = locs.unsqueeze(dim=2)

    event_time = event_time.to(device)  # (B, Seq_len)
    event_time_origin = event_time_origin.to(device)  # (B, Seq_len)
    event_locid = locs.to(device)  # (B, Seq_len, 2)
    event_loc_emb = d2c.embed(event_locid).to(device)

    enc_out, mask = transformer(event_loc_emb, event_time_origin)  # (B, Seq_len, dim), (B, Seq_len, 1)

    enc_out_non_mask = []
    event_time_non_mask = []
    event_loc_id_non_mask = []
    event_loc_emb_non_mask = []
    lengths = []
    for index in range(mask.shape[0]):
        length = int(sum(mask[index]).item())
        lengths.append(length - 1)
        if length > 1:
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length - 1]]
            event_time_non_mask += [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
            event_loc_emb_non_mask += [i.unsqueeze(dim=0) for i in event_loc_emb[index][1:length]]
            event_loc_id_non_mask += [i.unsqueeze(dim=0) for i in event_locid[index][1:length]]

    assert sum(lengths) == len(enc_out_non_mask)
    enc_out_non_mask = torch.cat(enc_out_non_mask, dim=0)  # (B*Seq_len-1, dim)
    event_time_non_mask = torch.cat(event_time_non_mask, dim=0)  # (B*Seq_len-1)
    event_loc_emb_non_mask = torch.cat(event_loc_emb_non_mask, dim=0)  # (B*Seq_len-1)
    event_loc_id_non_mask = torch.cat(event_loc_id_non_mask, dim=0)

    event_time_non_mask = event_time_non_mask.reshape(-1, 1, 1)  # (B*Seq_len-1, 1, 1)
    event_loc_emb_non_mask = event_loc_emb_non_mask.reshape(-1, 1, opt.loc_emb_dim)  # (B*Seq_len-1, 1, 2)
    event_loc_id_non_mask = event_loc_id_non_mask.reshape(-1, 1, 1)
    enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0], 1, -1)  # (B*Seq_len-1, 1, dim)

    return event_time_non_mask, event_loc_emb_non_mask, event_loc_id_non_mask, enc_out_non_mask, lengths


def Batch2toModel4sample(batch, transformer, d2c):
    event_time_origin, event_time, locs = map(lambda x: x.to(device), batch)
    event_loc = locs.unsqueeze(dim=2)

    event_time = event_time.to(device)  # (B, Seq_len)
    event_time_origin = event_time_origin.to(device)  # (B, Seq_len)
    event_locid = locs.to(device)  # (B, Seq_len, 2)
    event_loc_emb = d2c.embed(event_locid).to(device)

    enc_out, mask = transformer(event_loc_emb, event_time_origin)  # (B, Seq_len, dim), (B, Seq_len, 1)

    enc_out = enc_out[:, -1, :].unsqueeze(1)  # get the cond for the whole seq
    enc_out_non_mask = []
    for index in range(mask.shape[0]):
        enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index]]

    enc_out_non_mask = torch.cat(enc_out_non_mask, dim=0)  # (B, Seq_len-1, dim)
    enc_out_non_mask = enc_out_non_mask.reshape(enc_out.shape[0] * enc_out.shape[1], 1, -1)  # (B*Seq_len-1, 1, dim)

    return enc_out_non_mask


def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current + 1) / epoch_num


def reverse_normalization(norm_list, MAX, MIN):
    reversed_list = []
    for batch in norm_list:
        new_batch = []
        for sequence in batch:

            new_sequence = [(sequence[i] * (MAX[i] - MIN[i])) + MIN[i] for i in range(4)]
            new_batch.append(new_sequence)
        reversed_list.append(new_batch)
    return reversed_list


def generate_samples(Model, batch_size, seq_len, generate_num, first_checkin, MAX, MIN):
    trajectories = [[first_checkin + []] for _ in range(generate_num)]

    for _ in range(seq_len):
        current_traj = []
        data_loader = get_dataloader(trajectories, batch_size, D=opt.dim)
        for batch in data_loader:
            enc_out_non_mask = Batch2toModel4sample(batch, Model.transformer)
            sampled_seq = Model.diffusion.sample(batch_size = enc_out_non_mask.shape[0],cond=enc_out_non_mask).cpu()

            # append
            batch = [t.unsqueeze(-1) for t in batch]
            trajectory_tensor = torch.cat(batch, dim=-1)
            last_timestamps = trajectory_tensor[:, -1, 0].unsqueeze(1)  # Shape (B, 1)
            predicted_intervals = sampled_seq[:, :, 0] * (MAX[1] - MIN[1]) + MIN[1]  # Shape (B, 1)
            new_timestamps = last_timestamps + predicted_intervals  # Shape (B, 1)

            new_predicted_tensor = torch.cat([new_timestamps.unsqueeze(-1), sampled_seq], dim=2)  # Shape (B, 1, 4)
            combined_tensor = torch.cat([trajectory_tensor, new_predicted_tensor], dim=1)  # Shape (B, S+1, 4)
            current_traj += combined_tensor.tolist()
        trajectories = current_traj

    # recover
    trajectories = reverse_normalization(trajectories, MAX, MIN)
    trajectory_list = [[[sequence[0]] + sequence[2:] for sequence in batch] for batch in trajectories]
    return trajectory_list


def generate_dis_samples(Model, batch_size, seq_len, generate_num, first_checkin, max_interval, min_interval):
    # 1. sample first check in
    trajectories = [[random.choice(first_checkin) + []] for _ in range(generate_num)]

    for _ in range(seq_len):
        current_traj = []
        data_loader = get_dataloader(trajectories, batch_size, D=opt.dim)
        for batch in data_loader:
            enc_out_non_mask = Batch2toModel4sample(batch, Model.transformer, Model.d2c)
            sampled_seq = Model.diffusion.sample(batch_size = enc_out_non_mask.shape[0],cond=enc_out_non_mask)
            pred_embed = sampled_seq[:, :, 1:].squeeze(1)
            sampled_locid = Model.d2c.round(pred_embed).unsqueeze(1).unsqueeze(2)

            # append
            batch = [t.unsqueeze(-1) for t in batch]
            trajectory_tensor = torch.cat(batch, dim=-1)
            last_timestamps = trajectory_tensor[:, -1, 0].unsqueeze(1)  # Shape (B, 1)
            # predicted_intervals = sampled_seq[:, :, 0] * (max_interval - min_interval) + min_interval  # Shape (B, 1)
            last_interval = trajectory_tensor[:, -1, 1].unsqueeze(1)
            new_timestamps = last_timestamps + last_interval  # Shape (B, 1)

            f1 = new_timestamps.unsqueeze(-1).to(device)
            f2 = sampled_seq[:, :, 0].unsqueeze(-1).to(device)
            f3 = sampled_locid.to(device)

            new_predicted_tensor = torch.cat([f1, f2, f3], dim=2)  # Shape (B, 1, 4)
            combined_tensor = torch.cat([trajectory_tensor.to(device), new_predicted_tensor], dim=1)  # Shape (B, S+1, 4)
            current_traj += combined_tensor.cpu().tolist()
        trajectories = current_traj

    # recover
    # trajectories = reverse_normalization(trajectories, MAX, MIN)
    trajectory_list = [[[sequence[0]] + sequence[2:] for sequence in batch] for batch in trajectories]
    return trajectory_list


if __name__ == "__main__":
    print('dataset:{}'.format(opt.dataset))
    setup_init(opt)
    trainsize = 0.9
    logdir = "./02_01_24/logs/{}_timesteps_{}".format(opt.dataset, opt.timesteps)
    model_path = './02_01_24/ModelSave/dataset_{}_timesteps_{}/'.format(opt.dataset, opt.timesteps)
    result_path = './02_01_24/TrajectoryResults/dataset_{}_timesteps_{}/'.format(opt.dataset, opt.timesteps)

    if not os.path.exists('./ModelSave'):
        os.mkdir('./ModelSave')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)

    print('Model:{}'.format(opt.dataset))

    denoising_net = ST_Diffusion(
        n_steps=opt.timesteps,
        dim=1 + opt.loc_emb_dim,
        condition=True,
        cond_dim=64
    ).to(device)

    diffusion = GaussianDiffusion_ST(
        denoising_net,
        loss_type=opt.loss_type,
        seq_length=1 + opt.loc_emb_dim,
        timesteps=opt.timesteps,
        sampling_timesteps=opt.samplingsteps,
        objective=opt.objective,
        beta_schedule=opt.beta_schedule
    ).to(device)

    transformer = HEM(
        d_model=64,
        d_rnn=256,
        d_inner=128,
        n_layers=4,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        device=device,
        loc_dim=opt.loc_emb_dim,
        CosSin=True
    ).to(device)

    print('Dataloader:{}'.format(opt.dataset))

    dataset = FS_Dataset(opt.dataset, 0)
    gps = dataset.GPS  # Replace with your actual GPS data file
    traj = dataset.trajectories

    d2c = D2C(
        traj,
        gps,
        device=device,
        hidden_dim=16,
        embedding_dim=opt.loc_emb_dim,
        top_k=8,
        graph_d_path='graph/transition_graph_NYC.pt',
        graph_t_path='graph/distance_graph_NYC.pt',
        embedding_path='graph/l_embeddings_NYC.pt'
    ).to(device)

    Model = Model_all(d2c, transformer, diffusion)

    trainloader, valloader, testloader = split_list(traj, 0.9)

    warmup_steps = 5

    # training
    optimizer = AdamW(Model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_test = 1e20
    max_pred_rate = 0.5
    pred_type = 'sample_detach'

    for i in range(torch.cuda.device_count()):
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu_properties.name}, Capability: {gpu_properties.major}.{gpu_properties.minor}")\

    pre_train_embed = True
    if pre_train_embed:
        Model.d2c.pre_train(0.01, 200)

    for itr in range(opt.total_epochs):

        print('epoch:{}'.format(itr))
        # Val and test
        if itr % 200 == 0:
            print('Evaluate!')

            samples = generate_dis_samples(Model, opt.batch_size, opt.seq_len,
                                           opt.generate_num, dataset.first_checkins, dataset.max_interval, dataset.min_interval)
            # recover

            with open(result_path + str(itr) + '.pkl', 'wb') as f:
                pickle.dump(samples, f)

            # save model
            torch.save(Model.state_dict(), model_path + 'model_{}.pkl'.format(itr))

        # Train
        if itr < warmup_steps:
            for param_group in optimizer.param_groups:
                lr = LR_warmup(1e-3, warmup_steps, itr)
                pred_rate = 0
                param_group["lr"] = lr

        else:
            for param_group in optimizer.param_groups:
                lr = 1e-3 - (1e-3 - 5e-5) * (itr - warmup_steps) / opt.total_epochs
                pred_rate = max_pred_rate * (itr - warmup_steps) / opt.total_epochs
                param_group["lr"] = lr


        Model.train()

        loss_all, vb_all, vb_temporal_all, vb_spatial_all, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
        for batch in trainloader:
            event_time_non_mask, event_loc_emb_non_mask, event_loc_id_non_mask, enc_out_non_mask, lengths \
                = DisBatch2toModel(batch, Model.transformer, Model.d2c)
            if itr > warmup_steps:
                if pred_type == 'sample_detach':
                    sampled_seq = Model.diffusion.sample(batch_size=enc_out_non_mask.shape[0],
                                                         cond=enc_out_non_mask).detach()
                else:
                    sampled_seq = Model.diffusion.sample(batch_size=enc_out_non_mask.shape[0],
                                                         cond=enc_out_non_mask)

                pred_temporal = sampled_seq[:, :, 0].squeeze(1)
                pred_lat = sampled_seq[:, :, 1].squeeze(1)
                pred_long = sampled_seq[:, :, 2].squeeze(1)

                max_interval = 24 * 60 * 7
                pred_temporal = separate_and_pad_tensor(pred_temporal, lengths, batch[0].shape[1], 1e-10)
                pred_lat = separate_and_pad_tensor(pred_lat, lengths, batch[0].shape[1], 0.5)
                pred_long = separate_and_pad_tensor(pred_long, lengths, batch[0].shape[1], 0.5)

                original_interval = batch[1].clone()
                # replace lat and long
                selective_replace_in_tuples((batch[1], batch[2], batch[3]),
                                                   (pred_temporal, pred_lat, pred_long), pred_rate)

                res, update_success_count, update_fail_count = (
                    update_event_and_interval_times(batch[0], original_interval * max_interval, batch[1] * max_interval))

                new_batch = (res, batch[1], batch[2], batch[3])
                _, _, enc_out_non_mask, lengths = Batch2toModel(new_batch, Model.transformer)

            loss, pred_x_start = Model.diffusion(torch.cat((event_time_non_mask, event_loc_emb_non_mask), dim=-1), enc_out_non_mask)
            loss += Model.d2c(pred_x_start[:,:,1:].squeeze(1), event_loc_id_non_mask.view(event_loc_id_non_mask.shape[0]))

            optimizer.zero_grad()
            loss.backward()

            loss_all += loss.item() * event_time_non_mask.shape[0]
            vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(
                torch.cat((event_time_non_mask, event_loc_emb_non_mask), dim=-1), enc_out_non_mask)

            vb_all += vb
            vb_temporal_all += vb_temporal
            vb_spatial_all += vb_spatial

            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step()

            step += 1

            total_num += event_time_non_mask.shape[0]

        if opt.cuda_id != 'cpu':
            with torch.cuda.device("cuda"):
                torch.cuda.empty_cache()

        print("Training/loss_epoch " + str(itr) + ' : ' + str(loss_all / total_num))
        print("Training/NLL_epoch " + str(itr) + ' : ' + str(vb_all / total_num))
        print("Training/NLL_temporal_epoch " + str(itr) + ' : ' + str(vb_temporal_all / total_num))
        print("Training/NLL_spatial_epoch " + str(itr) + ' : ' + str(vb_spatial_all / total_num))

