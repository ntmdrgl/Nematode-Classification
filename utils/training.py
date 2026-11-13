import torch

def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    model.train()
    running_loss = 0.0
    sample_count = 0

    for inputs, labels in train_loader:
        # use device
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # go forward through model and find loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backpropagation and optimize
        optimizer.zero_grad() # zero gradient buffers
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients

        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Training loader produced no samples.")

    epoch_loss = running_loss / sample_count
    return epoch_loss


def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    num_classes
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # per-class tracking
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            batch_size = labels.size(0)
            total += batch_size
            running_loss += loss.item() * batch_size

            labels_cpu = labels.cpu().tolist()
            predicted_cpu = predicted.cpu().tolist()

            # update per-class stats
            for label, pred in zip(labels_cpu, predicted_cpu):
                class_total[label] += 1
                if pred == label:
                    correct += 1
                    class_correct[label] += 1

    if total == 0:
        raise ValueError("Evaluation loader produced no samples.")

    # overall accuracy
    overall_accuracy = 100 * correct / total
    epoch_loss = running_loss / total

    # per-class accuracy
    class_accuracy = [
        100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(num_classes)
    ]

    return epoch_loss, overall_accuracy, class_accuracy

def predict(model, input_loader, device):
    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in input_loader:
            inputs = inputs.to(device, non_blocking=True)

            outputs = model(inputs) # (batch_size, num_classes)
            probabilities = torch.softmax(outputs, dim=1)

            all_probabilities.append(probabilities)
            all_labels.append(labels)

    all_probabilities = torch.cat(all_probabilities, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_probabilities, all_labels