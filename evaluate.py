from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy().transpose(1,2,0)
    y_pred = y_pred.detach().cpu().numpy().transpose(1,2,0)
    return ssim(y_true, y_pred, channel_axis=2), psnr(y_true, y_pred)
