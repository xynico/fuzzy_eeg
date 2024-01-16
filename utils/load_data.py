import pickle
import torch
from sklearn.model_selection import train_test_split
from itertools import product
from torch.utils.data import Dataset
def flatten_dict(d):
    data = []
    label= []
    pair_label=[]
    for k, v in d.items():
        if isinstance(v, dict):
            flattened_tensor, flattened_label,flattened_pair=flatten_dict(v)
            #(l_m, B, H, W, ), (l_m, B,1)
            data.append(flattened_tensor)
            label.append(flattened_label)
            pair_label.append(flattened_pair)
        else:
            # print("k",type(k))
            sk=int(k.replace("-", ""))
            sub1=torch.from_numpy(v[0])
            sub2=torch.from_numpy(v[1])
            sub_tensor=torch.cat((sub1,sub2),dim=0)
            sub_label=[0]*sub1.shape[0] + [1]*sub2.shape[0]
            sub_label=torch.tensor(sub_label)
            # print("sub_label",sub_label.shape)
            pair=[sk]*sub1.shape[0] + [sk]*sub2.shape[0]
            pair=torch.tensor(pair)
            # print("pair_label,",len(pair_label))
            data.append(sub_tensor)
            label.append(sub_label)
            pair_label.append(pair)
    f_tensor=torch.cat(data, dim=0)
    f_label = torch.cat(label, dim=0)
    pair_label = torch.cat(pair_label, dim=0)
    return f_tensor, f_label,pair_label

def min_max_norm(in_tensor):
    min_val = in_tensor.min()
    range_val = in_tensor.max() - min_val
    normalized_tensor = (in_tensor - min_val) / range_val
    return normalized_tensor
def get_data_label(file_path = '/projects/CIBCIGroup/00DataUploading/Xiaowei/Friend_fNIRS/step1_PPCS_0.01_0.2PictureRating_data.pkl'
):
    with open(file_path, 'rb') as file:
        # Load the dataset from the file
        dataset = pickle.load(file)
    tensor_data, tensor_labels,tensor_pair_label = flatten_dict(dataset)
    return tensor_data, tensor_labels,tensor_pair_label
def find_unique(label_all):
    # Finding unique values
    unique_values = torch.unique(label_all)

    # Finding indices for each unique value
    indices_per_unique_value = {val.item(): (label_all == val).nonzero(as_tuple=True)[0].tolist() for val in
                                unique_values}
    return indices_per_unique_value
def slice_data(data, uniq):
    slices = []

    # Iterate over each unique value and its indices
    for _, indices in uniq.items():
        # Slicing tensor2 using the indices
        slice_ = data[indices]
        # Append the slice to the list
        slices.append(slice_)
    return slices
def process_data(seed=0):
    data, label_sub, label_all=get_data_label()
    data=min_max_norm(data)
    # Finding unique values
    unique_comb=find_unique(label_all)
    unique_values = torch.unique(label_all)
    sliced_data=slice_data(data,unique_comb)
    sliced_sub=slice_data(label_sub,unique_comb)
    # Finding indices for each unique value
    # Concatenating all slices
    bad_exp=[]
    train_data=[]
    test_data=[]
    for i in range (len(sliced_data)):
        features=sliced_data[i]
        labels=sliced_sub[i]
        # Separate features based on labels
        features_0 = features[labels == 0]
        features_1 = features[labels == 1]
        if features_0.shape[0]<5 or features_1.shape[0]<5:
            bad_exp.append(i)
        else:
            # Generate all combinations
            torch.manual_seed(seed)

            # Splitting the data into training and testing sets (70% train, 30% test)
            features_0_train, features_0_test = train_test_split(features_0, test_size=0.3)
            features_1_train, features_1_test = train_test_split(features_1, test_size=0.3)

            combinations_train = product(features_0_train, features_1_train)
            combinations_test = product(features_0_test, features_1_test)
            # print("features_0, features_1", features_0.shape, features_1.shape)

            # Convert combinations to tensor and reshape
            combinations_train = torch.stack([torch.stack([f0, f1]) for f0, f1 in combinations_train])
            combinations_test = torch.stack([torch.stack([f0, f1]) for f0, f1 in combinations_test])
            train_data.append(combinations_train)
            test_data.append(combinations_test)
    valid_GT = list(unique_comb.keys())
    filtered_GT = [key for index, key in enumerate(valid_GT) if index not in bad_exp]
    return train_data,test_data,filtered_GT
def create_train_test_sets(seed=0):
    train_data, test_data, filtered_GT = process_data(seed=seed)
    train_set = EEGDataset(train_data, filtered_GT)
    test_set=EEGDataset(test_data, filtered_GT)
    return train_set,test_set
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        # Repeat each element in the second list 'B' times
        repeated_list = [item for sublist in [[x] * data[i].shape[0] for i, x in enumerate(labels)] for
                         item in sublist]
        repeated_list = [int(str(item)[1]) for item in repeated_list]
        # Concatenate all tensors in the first list
        concatenated_tensors = torch.cat(data, dim=0)
        self.data = concatenated_tensors
        self.labels = torch.tensor(repeated_list)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

