import torch

num_segments = tensor_data.shape[1] // segment_size

print(f"Segment size: {segment_size}")
print(f"Number of segments per sample: {num_segments}\n")

print("Before segmentation:")
print("  train_data shape:", splits[0]['train_data'].shape)
print("  train_labels shape:", splits[0]['train_labels'].shape)
print("  val_data shape:", splits[0]['val_data'].shape)
print("  val_labels shape:", splits[0]['val_labels'].shape)
print("  test_data shape:", splits[0]['test_data'].shape)
print("  test_labels shape:", splits[0]['test_labels'].shape)

for fold in splits:
    train_data = fold['train_data']
    train_labels = fold['train_labels']
    test_data = fold['test_data']
    val_data = fold['val_data']
    val_labels = fold['val_labels']
    test_data = fold['test_data']
    test_labels = fold['test_labels']
    
    segment_train_data = []
    segment_train_labels = []

    for data, label in zip(train_data, train_labels):
        num_segments = data.shape[0] // segment_size 

        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            
            segment_train_data.append(data[start:end])
            segment_train_labels.append(label)
    
    segment_train_data = torch.stack(segment_train_data)
    segment_train_labels = torch.tensor(segment_train_labels)

    segment_val_data = []
    segment_val_labels = []

    for data, label in zip(val_data, val_labels):
        num_segments = data.shape[0] // segment_size 

        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            
            segment_val_data.append(data[start:end])
            segment_val_labels.append(label)

    segment_val_data = torch.stack(segment_val_data)
    segment_val_labels = torch.tensor(segment_val_labels)

    segment_test_data = []
    segment_test_labels = []

    for data, label in zip(test_data, test_labels):
        num_segments = data.shape[0] // segment_size 

        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            
            segment_test_data.append(data[start:end])
            segment_test_labels.append(label)

    segment_test_data = torch.stack(segment_test_data)
    segment_test_labels = torch.tensor(segment_test_labels)

    fold['train_data'] = segment_train_data
    fold['train_labels'] = segment_train_labels
    fold['val_data'] = segment_val_data
    fold['val_labels'] = segment_val_labels
    fold['test_data'] = segment_test_data
    fold['test_labels'] = segment_test_labels

print("After segmentation:")
print("  train_data shape:", splits[0]['train_data'].shape)
print("  train_labels shape:", splits[0]['train_labels'].shape)
print("  val_data shape:", splits[0]['val_data'].shape)
print("  val_labels shape:", splits[0]['val_labels'].shape)
print("  test_data shape:", splits[0]['test_data'].shape)
print("  test_labels shape:", splits[0]['test_labels'].shape)