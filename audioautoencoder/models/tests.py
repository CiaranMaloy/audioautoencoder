import unittest
import torch
from torchsummary import summary
from models.UNetDenoisingAutoencoder import *  # Import your model

class TestAutoencoder(unittest.TestCase):
    def setUp(self):
        self.model = UNetDenoisingAutoencoder()
        self.input_channels = 2
        self.input_height = 1025
        self.input_width = 175
        self.batch_size = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, UNetDenoisingAutoencoder, "Model initialization failed")

    def test_forward_pass(self):
        x = torch.randn(self.batch_size, self.input_channels, self.input_height, self.input_width, device=self.device)
        output = self.model(x)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.input_channels, self.input_height, self.input_width),
            f"Expected output shape {(self.batch_size, self.input_channels, self.input_height, self.input_width)}, but got {output.shape}"
        )

    def test_model_summary(self):
        try:
            summary(self.model, input_size=(self.input_channels, self.input_height, self.input_width))
        except Exception as e:
            self.fail(f"Model summary failed: {str(e)}")

# This allows running tests externally
def suite(Model):
    test_suite = unittest.TestLoader().loadTestsFromTestCase(Model)
    return test_suite

# runner
class TestRunner:
    def __init__(self):
        self.runner = unittest.TextTestRunner()

    def run(self, Model):
        print("Running autoencoder tests...")
        self.runner.run(suite(Model))

if __name__ == "__main__":
    runner = TestRunner(TestAutoencoder)
    runner.run()
