"""
Camera-based ResidualController variant.
Uses camera1/camera2 RGB images as observation instead of tactile features.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score

class CameraResidualController(pl.LightningModule):
    """
    Camera-based model: RGB image (H, W, 3) -> action sequence (H, 7).
    Uses CNN for visual feature extraction.
    """
    def __init__(self, action_dim=7, obs_dim=256, horizon=16, lr=1e-4,
                 camera_name='camera1', img_size=(480, 640)):
        super().__init__()
        self.save_hyperparameters()
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.lr = float(lr)
        self.camera_name = camera_name
        self.img_size = img_size
        
        # CNN feature extractor for RGB images
        # Input: (B, 3, H, W) where H=480, W=640
        self.feature_extractor = nn.Sequential(
            # Conv1: (3, 480, 640) -> (32, 240, 320)
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, 120, 160)
            
            # Conv2: (32, 120, 160) -> (64, 60, 80)
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (64, 30, 40)
            
            # Conv3: (64, 30, 40) -> (128, 15, 20)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (128, 7, 10)
            
            # Conv4: (128, 7, 10) -> (256, 4, 5)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (256, 1, 1)
            nn.Flatten()  # -> (256,)
        )
        
        # Project CNN features to obs_dim if needed
        self.feature_to_obs = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, obs_dim)
        ) if obs_dim != 256 else nn.Identity()
        
        # Decision network: obs -> action sequence
        self.model = nn.Sequential(
            nn.Linear(horizon * obs_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, horizon * action_dim)
        )
        
        self.loss_fn = nn.MSELoss()
    
    def extract_features(self, image):
        """
        Extract features from RGB image.
        image: (B, H, W, 3) or (B, 3, H, W), uint8 or float
        Returns: (B, obs_dim)
        """
        # Ensure (B, 3, H, W) format
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)  # (B, H, W, 3) -> (B, 3, H, W)
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image.float() / 255.0
        
        # Extract CNN features
        cnn_features = self.feature_extractor(image)  # (B, 256)
        obs_token = self.feature_to_obs(cnn_features)  # (B, obs_dim)
        return obs_token
    
    def forward(self, obs):
        """
        obs: (B, H, W, 3) or (B, 3, H, W) RGB image
        Returns: (B, horizon, action_dim) predicted actions
        """
        obs_token = self.extract_features(obs)  # (B, obs_dim)
        
        # Repeat for horizon
        obs_seq = obs_token.unsqueeze(1).repeat(1, self.horizon, 1)  # (B, H, obs_dim)
        
        # Flatten and predict
        batch_size = obs_seq.shape[0]
        flattened = obs_seq.reshape(batch_size, -1)  # (B, H*obs_dim)
        pred_flat = self.model(flattened)  # (B, H*action_dim)
        pred_actions = pred_flat.view(batch_size, self.horizon, self.action_dim)
        
        return pred_actions
    
    def _compute_r2(self, pred, target):
        pred_flat = pred.detach().cpu().numpy().flatten()
        target_flat = target.detach().cpu().numpy().flatten()
        try:
            return r2_score(target_flat, pred_flat)
        except:
            return 0.0
    
    def training_step(self, batch, batch_idx):
        obs, actions = batch
        pred_actions = self(obs)
        loss = self.loss_fn(pred_actions, actions)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        obs, actions = batch
        pred_actions = self(obs)
        loss = self.loss_fn(pred_actions, actions)
        r2 = self._compute_r2(pred_actions, actions)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_r2', r2, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        obs, actions = batch
        pred_actions = self(obs)
        loss = self.loss_fn(pred_actions, actions)
        r2 = self._compute_r2(pred_actions, actions)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_r2', r2, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
