import torch

def erb_filter_bank_torch(sr=44100, n_fft=2048, n_bands=24, device='cpu'):
    """
    Generate an ERB filter bank in PyTorch.

    Args:
        sr (int): Sampling rate of the audio.
        n_fft (int): Number of FFT points.
        n_bands (int): Number of ERB bands.
        device (str): Device for computation ('cpu' or 'cuda').

    Returns:
        torch.Tensor: ERB filter bank matrix of shape (n_bands, n_fft // 2 + 1).
    """
    # Generate ERB center frequencies using mel scale approximation
    mel_points = torch.linspace(0, sr / 2, n_bands + 2, device=device)
    erb_filters = torch.zeros((n_bands, n_fft // 2), device=device)

    for i in range(1, n_bands + 1):
        left = mel_points[i - 1]
        center = mel_points[i]
        right = mel_points[i + 1]

        for j, freq in enumerate(torch.linspace(0, sr / 2, n_fft // 2, device=device)):
            if left < freq < center:
                erb_filters[i - 1, j] = (freq - left) / (center - left)
            elif center < freq < right:
                erb_filters[i - 1, j] = (right - freq) / (right - center)

    return erb_filters


def compute_erb_features_torch(output, erb_filters):
    """
    Compute ERB features in PyTorch for GPU compatibility.

    Args:
        audio (torch.Tensor): Input audio signal (batch_size, signal_length).
        sr (int): Sampling rate of the audio signal.
        erb_filters (torch.Tensor): ERB filter bank matrix.
        n_fft (int): Number of FFT points.
        hop_length (int): Hop length for STFT.

    Returns:
        torch.Tensor: ERB feature matrix of shape (batch_size, n_bands, time_frames).
    """
    # Compute STFT
    #stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length,
    #                  win_length=n_fft, return_complex=True)
    #spectrogram = torch.abs(stft)  # Magnitude

    # Apply ERB filter bank
    #print(erb_filters)
    erb_features = torch.matmul(erb_filters, output)

    #print('Mean')
    #print(np.mean(erb_features.detach().cpu().numpy()[0][0]))
    #plt.imshow(erb_features.detach().cpu().numpy()[0][0])
    #plt.show()

    return erb_features


class ERBLoss(torch.nn.Module):
    """
    ERB Perceptual Loss combined with MSE Loss.
    """
    def __init__(self, alpha=0.2, device='cuda'):
        """
        Initialize the ERBLoss module.

        Args:
            sr (int): Sampling rate of the audio.
            n_fft (int): Number of FFT points.
            hop_length (int): Hop length for STFT.
            n_bands (int): Number of ERB bands.
            alpha (float): Weighting factor for the perceptual loss.
            device (str): Device for computation ('cpu' or 'cuda').
        """
        super().__init__()
        self.alpha = alpha
        self.device = device
        self.mse_loss = torch.nn.MSELoss()
        self.erb_filters = erb_filter_bank_torch(device=device)

    def forward(self, predicted, target):
        """
        Compute the combined ERB and MSE loss.

        Args:
            predicted (torch.Tensor): Predicted audio signal of shape (batch_size, signal_length).
            target (torch.Tensor): Target audio signal of shape (batch_size, signal_length).

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Compute ERB features for both predicted and target signals
        pred_erb = compute_erb_features_torch(predicted, self.erb_filters)
        target_erb = compute_erb_features_torch(target, self.erb_filters)

        # Compute perceptual loss
        perceptual_loss = self.mse_loss(pred_erb, target_erb)
        #print(f'Perceptual loss: {perceptual_loss}')

        # Compute standard MSE loss
        mse_loss = self.mse_loss(predicted, target)
        #print(f'MSE loss: {mse_loss}')

        # Combine losses with weighting factor alpha
        combined_loss = self.alpha * perceptual_loss + (1 - self.alpha) * mse_loss

        return combined_loss
    
import torch
import unittest

class TestERBLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.input_channels = 3
        self.input_height = 1024
        self.input_width = 84  # 1 second of audio at 16kHz

        self.n_bands = 24
        self.alpha = 0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)

        # Create random signals for predicted and target
        self.predicted = torch.randn(self.batch_size, self.input_channels, self.input_height, self.input_width, device=self.device, requires_grad=True)
        self.target = torch.randn(self.batch_size, self.input_channels, self.input_height, self.input_width, device=self.device, requires_grad=True)

        # Initialize the ERBLoss
        self.loss_fn = ERBLoss(
            alpha=self.alpha,
            device=self.device
        )

    def test_loss_computation(self):
        """
        Test if the ERBLoss computes a valid loss value.
        """
        loss = self.loss_fn(self.predicted, self.target)
        self.assertIsInstance(loss, torch.Tensor, "Loss should be a torch.Tensor.")
        self.assertGreater(loss.item(), 0, "Loss value should be positive.")

    def test_gradient_flow(self):
        """
        Test if gradients flow through the ERBLoss correctly.
        """
        loss = self.loss_fn(self.predicted, self.target)

        # Debugging gradient flow
        print("Predicted requires grad:", self.predicted.requires_grad)
        print("Target requires grad:", self.target.requires_grad)

        pred_erb = compute_erb_features_torch(self.predicted, self.loss_fn.erb_filters)
        target_erb = compute_erb_features_torch(self.target, self.loss_fn.erb_filters)

        print("Pred ERB requires grad:", pred_erb.requires_grad)
        print("Target ERB requires grad:", target_erb.requires_grad)

        loss.backward()

    def test_gradient_flow_2(self):
        # Ensure tensors have requires_grad set to True
        self.predicted = self.predicted.requires_grad_()
        self.target = self.target.requires_grad_()

        loss = self.loss_fn(self.predicted, self.target)
        self.assertTrue(loss.requires_grad)  # Ensure the loss has requires_grad

        # Run backward pass and check if gradients flow correctly
        loss.backward()

        # Check if gradients are flowing through the predicted and target tensors
        self.assertTrue(self.predicted.grad is not None)
        self.assertTrue(self.target.grad is not None)


    def test_device_compatibility(self):
        """
        Test if the ERBLoss works correctly on both CPU and CUDA (if available).
        """
        # Test on CPU
        self.predicted = self.predicted.to('cpu')
        self.target = self.target.to('cpu')
        loss_cpu = self.loss_fn(self.predicted, self.target)

        # Test on CUDA (if available)
        if torch.cuda.is_available():
            self.predicted = self.predicted.to('cuda')
            self.target = self.target.to('cuda')
            self.loss_fn = self.loss_fn.to('cuda')  # Move loss function to CUDA
            loss_cuda = self.loss_fn(self.predicted, self.target)
            self.assertAlmostEqual(
                loss_cpu.item(), loss_cuda.item(), places=5,
                msg="Loss values on CPU and CUDA should be nearly identical."
            )

    def test_alpha_weighting(self):
        """
        Test if the alpha parameter affects the weighting of perceptual and MSE loss.
        """
        # Create a new loss function with different alpha values
        loss_fn_alpha_1 = ERBLoss(
            alpha=1.0,
            device=self.device
        )
        loss_fn_alpha_0 = ERBLoss(
            alpha=0.0,
            device=self.device
        )

        # Compute losses
        loss_alpha_1 = loss_fn_alpha_1(self.predicted, self.target)
        loss_alpha_0 = loss_fn_alpha_0(self.predicted, self.target)

        # Check if the losses match expected behavior
        perceptual_loss = loss_fn_alpha_1.mse_loss(
            compute_erb_features_torch(self.predicted, loss_fn_alpha_1.erb_filters),
            compute_erb_features_torch(self.target, loss_fn_alpha_1.erb_filters)
        )
        mse_loss = loss_fn_alpha_0.mse_loss(self.predicted, self.target)

        self.assertAlmostEqual(loss_alpha_1.item(), perceptual_loss.item(), places=5,
                               msg="Alpha=1.0 should result in only perceptual loss.")
        self.assertAlmostEqual(loss_alpha_0.item(), mse_loss.item(), places=5,
                               msg="Alpha=0.0 should result in only MSE loss.")

if False:
  if __name__ == "__main__":
      unittest.main(argv=['first-arg-is-ignored'], exit=False)
