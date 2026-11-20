"""Lab 14: Vanilla RNN with a manual forward and backward pass (BPTT)."""
import numpy as np

np.random.seed(14)
vin_num_samples = 200
vin_sequence_length = 6
vin_input_size = 1
vin_hidden_size = 8

# ---------- Toy sequence classification: target is 1 when sequence sum is greater than 3
vin_x = np.random.randint(0, 2, (vin_num_samples, vin_sequence_length, 1)).astype(float)
vin_y = (np.sum(vin_x, axis=1) > 3).astype(float)
vin_wxh = np.random.randn(vin_input_size, vin_hidden_size) * 0.2
vin_whh = np.random.randn(vin_hidden_size, vin_hidden_size) * 0.2
vin_why = np.random.randn(vin_hidden_size, 1) * 0.2
vin_bh = np.zeros((1, vin_hidden_size))
vin_by = np.zeros((1, 1))
vin_learning_rate = 0.05

def vin_sigmoid(vin_value):
    return 1.0 / (1.0 + np.exp(-vin_value))

for vin_epoch in range(100):
    vin_total_loss = 0.0
    for vin_sample in range(vin_num_samples):
        vin_states = [np.zeros((1, vin_hidden_size))]
        # ---------- Forward sequence one time step at a time
        for vin_time in range(vin_sequence_length):
            vin_current_x = vin_x[vin_sample, vin_time].reshape(1, 1)
            vin_current_h = np.tanh(
                np.dot(vin_current_x, vin_wxh)
                + np.dot(vin_states[-1], vin_whh)
                + vin_bh
            )
            vin_states.append(vin_current_h)
        vin_prediction = vin_sigmoid(np.dot(vin_states[-1], vin_why) + vin_by)
        vin_target = vin_y[vin_sample].reshape(1, 1)
        vin_epsilon = 1e-8
        vin_loss = -np.mean(vin_target * np.log(vin_prediction + vin_epsilon)
                            + (1 - vin_target) * np.log(1 - vin_prediction + vin_epsilon))
        vin_total_loss = vin_total_loss + vin_loss

        # ---------- Backpropagation through time
        vin_dwhy = np.dot(vin_states[-1].T, vin_prediction - vin_target)
        vin_dby = vin_prediction - vin_target
        vin_dh_next = np.dot(vin_prediction - vin_target, vin_why.T)
        vin_dwxh = np.zeros_like(vin_wxh)
        vin_dwhh = np.zeros_like(vin_whh)
        vin_dbh = np.zeros_like(vin_bh)
        for vin_time in range(vin_sequence_length - 1, -1, -1):
            vin_dh = vin_dh_next
            vin_dz = vin_dh * (1 - vin_states[vin_time + 1] ** 2)
            vin_current_x = vin_x[vin_sample, vin_time].reshape(1, 1)
            vin_dwxh = vin_dwxh + np.dot(vin_current_x.T, vin_dz)
            vin_dwhh = vin_dwhh + np.dot(vin_states[vin_time].T, vin_dz)
            vin_dbh = vin_dbh + vin_dz
            vin_dh_next = np.dot(vin_dz, vin_whh.T)

        # Clip gradients to reduce exploding gradients
        vin_dwxh = np.clip(vin_dwxh, -5, 5)
        vin_dwhh = np.clip(vin_dwhh, -5, 5)
        vin_dwhy = np.clip(vin_dwhy, -5, 5)
        vin_wxh = vin_wxh - vin_learning_rate * vin_dwxh
        vin_whh = vin_whh - vin_learning_rate * vin_dwhh
        vin_why = vin_why - vin_learning_rate * vin_dwhy
        vin_bh = vin_bh - vin_learning_rate * vin_dbh
        vin_by = vin_by - vin_learning_rate * vin_dby
    if vin_epoch % 10 == 0:
        print("Epoch", vin_epoch, "loss", vin_total_loss / vin_num_samples)

print("Example sequence:", vin_x[0].ravel())
print("Predicted probability:", vin_prediction.ravel())
print("Target:", vin_y[0].ravel())
