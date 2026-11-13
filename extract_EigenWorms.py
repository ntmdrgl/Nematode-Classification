from pathlib import Path

import numpy as np
from aeon.datasets import load_from_ts_file

raw = Path("data/EigenWorms")
output = Path("data")
output.mkdir(parents=True, exist_ok=True)

X_train, labels_train = load_from_ts_file(
    str(raw / "EigenWorms_TRAIN.ts")
)
X_test, labels_test = load_from_ts_file(
    str(raw / "EigenWorms_TEST.ts")
)

# Use the same label mapping for both splits.
classes = sorted(set(labels_train.tolist()))
class_to_id = {label: i for i, label in enumerate(classes)}

y_train = np.array([class_to_id[v] for v in labels_train], dtype=np.int64)
y_test = np.array([class_to_id[v] for v in labels_test], dtype=np.int64)

X_train = np.asarray(X_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)

assert X_train.shape[1:] == X_test.shape[1:] == (6, 17984)
assert np.isfinite(X_train).all() and np.isfinite(X_test).all()

np.savez_compressed(
    output / "eigenworms.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    classes=np.asarray(classes, dtype=str),
)

print("Train:", X_train.shape, y_train.shape)
print("Test: ", X_test.shape, y_test.shape)
print("Labels:", class_to_id)