import torch
import matplotlib.pyplot as plt
from pathlib import Path
from train.data_loader import get_dataloader
from models.convlstm import ConvLSTM_Predictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = Path("data_imgs/light_cloud")
loader = get_dataloader(data_dir, batch_size=1, seq_len=3, num_workers=0)

# load model
model = ConvLSTM_Predictor(input_dim=3, hidden_dim=32, kernel_size=(3,3), n_layers=2).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/convlstm_best.pth", map_location=DEVICE))
model.eval()

x, y = next(iter(loader))
x, y = x.to(DEVICE), y.to(DEVICE)

with torch.no_grad():
    y_pred = model(x)

x_img = x[0, -1].cpu().permute(1,2,0)
y_img = y[0].cpu().permute(1,2,0)
y_pred_img = y_pred[0].cpu().permute(1,2,0).clamp(0,1)

plt.figure(figsize=(10,3))
plt.subplot(1,3,1); plt.imshow(x_img); plt.title("Input (t)")
plt.subplot(1,3,2); plt.imshow(y_pred_img); plt.title("Predicted (t+1)")
plt.subplot(1,3,3); plt.imshow(y_img); plt.title("Ground Truth")
plt.tight_layout()
plt.savefig("results/final_prediction.png")
plt.show()
