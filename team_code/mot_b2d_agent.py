import os
import sys
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import imageio
import random
import sys
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

project_root = str(pathlib.Path(__file__).parent.parent.parent)
leaderboard_root = str(os.path.join(project_root, 'leaderboard'))
scenario_runner_root = str(os.path.join(project_root, 'scenario_runner'))
mot_dp_root = str(os.path.join(project_root, 'MoT-DP'))
carla_api_root = str(os.path.join(project_root.replace('Bench2Drive', 'carla'), 'PythonAPI', 'carla'))

for path in [project_root, leaderboard_root, scenario_runner_root, mot_dp_root, carla_api_root]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

sys.path = [str(p) for p in sys.path]

from leaderboard.autoagents import autonomous_agent
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy
from team_code.simlingo.nav_planner import RoutePlanner, LateralPIDController  
import team_code.simlingo.transfuser_utils as t_u  
from team_code.render import render, render_self_car, render_waypoints
from dataset.generate_lidar_bev_b2d import generate_lidar_bev_images
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import xml.etree.ElementTree as ET  
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  
# mot dependencies
project_root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
mot_dp_path = str(os.path.join(os.path.dirname(os.path.dirname(project_root)), 'MoT-DP'))
mot_path = str(os.path.join(mot_dp_path, 'mot'))
sys.path.append(mot_dp_path)
sys.path.append(mot_path)
sys.path = [str(p) for p in sys.path]

from transformers import HfArgumentParser
import json
from dataclasses import dataclass, field
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from PIL import Image
from safetensors.torch import load_file
import glob
from data.reasoning.data_utils import add_special_tokens
from mot.modeling.automotive import (
    AutoMoTConfig, AutoMoT,
    Qwen3VLTextConfig, Qwen3VLTextModel, Qwen3VLForConditionalGenerationMoT
)
from dataset.unified_carla_dataset import CARLAImageDataset
from policy.diffusion_dit_carla_policy import DiffusionDiTCarlaPolicy
from mot.evaluation.inference import InterleaveInferencer
from transformers import AutoTokenizer

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PLANNER_TYPE = os.environ.get('PLANNER_TYPE', None)
EARTH_RADIUS_EQUA = 6378137.0
USE_UKF = True  # Enable Unscented Kalman Filter for GPS/compass smoothing

# dp utils
def get_entry_point():
	return 'MOTAgent'

def create_carla_config(config_path=None):
    if config_path is None:
        config_path = "/home/wang/Project/MoT-DP/config/pdm_local.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_best_model(checkpoint_path, config, device):
    print(f"Loading best model from: {checkpoint_path}")
    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    action_stats = {
    'min': torch.tensor([-0.06638534367084503, -17.525903701782227]),
    'max': torch.tensor([74.04539489746094, 32.73622512817383]),
    'mean': torch.tensor([12.758530616760254, 0.354688435792923]),
    'std': torch.tensor([16.723825454711914, 2.5529885292053223]),
    }

    policy = DiffusionDiTCarlaPolicy(config, action_stats=action_stats).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  - Training Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    
    if 'val_metrics' in checkpoint:
        print(f"  - Validation Metrics:")
        for key, value in checkpoint['val_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value:.4f}")

    return policy

# display purpose only
class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        pygame.display.set_caption("CORL Agent")

    def run_interface(self, input_data):
        rgb = input_data['rgb']
        trajectory = input_data['predicted_trajectory']
        decision_1s = input_data['decision_1s']
        decision_2s = input_data['decision_2s']
        decision_3s = input_data['decision_3s']
        surface = np.zeros((600, 1200, 3),np.uint8)
        surface[:, :800] = rgb
        surface[:400,800:1200] = input_data['bev_traj']
        surface[440:600,1000:1200] = trajectory[0:160,:]
        surface = cv2.putText(surface, input_data['language_1'], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
        surface = cv2.putText(surface, input_data['language_2'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
        surface = cv2.putText(surface, input_data['control'], (20,540), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
        surface = cv2.putText(surface, input_data['speed'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)
        surface = cv2.putText(surface, input_data['time'], (20,500), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 1)

        # surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        # surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        # surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

        surface = cv2.putText(surface, 'Behavior Decision', (820,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 'Planned Trajectory', (1010,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, decision_1s, (820,480), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255), 2)
        surface = cv2.putText(surface, decision_2s, (820,510), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255), 2)
        surface = cv2.putText(surface, decision_3s, (820,540), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255), 2)

        surface[:150,198:202]=0
        surface[:150,323:327]=0
        surface[:150,473:477]=0
        surface[:150,598:602]=0
        surface[148:152, :200] = 0
        surface[148:152, 325:475] = 0
        surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255


        # display image
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()

# mot utils
@dataclass
class ModelArguments:
    model_path: str = field(
        default="/home/wang/Project/MoT-DP/checkpoints/mot",
        metadata={"help": "Path to the converted AutoMoT model checkpoint"}
    )
    qwen3vl_path: str = field(
        default="/home/wang/Project/MoT-DP/config/mot_config",
        metadata={"help": "Path to the Qwen3VL base model for config loading"}
    )
    max_latent_size: int = field(
        default=64,
        metadata={"help": "Maximum size of latent representations"}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Patch size for latent space processing"}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of patches per side for vision transformer"}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function for connector layers"}
    )
    mot_num_attention_heads: int = field(
        default=16,
        metadata={"help": "Number of attention heads for MoT attention components. Defaults to half of regular attention heads if not specified."}
    )
    mot_num_key_value_heads: int = field(
        default=4,
        metadata={"help": "Number of key-value heads for MoT attention components. Defaults to half of regular KV heads if not specified."}
    )
    mot_intermediate_size: int = field(
        default=4864,
        metadata={"help": "Intermediate size for MoT MLP components. Defaults to same as regular intermediate_size if not specified."}
    )
    reasoning_query_dim: int = field(
        default=588,
        metadata={"help": "Dimension of the reasoning query embedding."}
    )
    reasoning_query_max_num_tokens: int = field(
        default=8, #256, #64,
        metadata={"help": "Maximum number of tokens in the reasoning query."}
    )
    action_query_dim: int = field(
        default=588,
        metadata={"help": "Dimension of the action query embedding."}
    )
    action_query_tokens: int = field(
        default=1,
        metadata={"help": "Number of tokens in the action query."}
    )

@dataclass
class InferenceArguments:
    dataset_jsonl: str = field(
        default="b2d_data_val.jsonl",
        metadata={"help": "Path to the input dataset JSONL file"}
    )
    output_jsonl: str = field(
        default="",
        metadata={"help": "Path to the output JSONL file. If empty, will use input filename with .pred.jsonl suffix"}
    )
    base_path: str = field(
        default="/share-data/pdm_lite",
        metadata={"help": "Base path for resolving relative image paths. If empty, use current working directory"}
    )
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Enable visual generation capabilities"}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Enable visual understanding capabilities"}
    )
    max_num_tokens: int = field(
        default=16384,
        metadata={"help": "Maximum number of tokens for inference"}
    )
    start_idx: int = field(
        default=0,
        metadata={"help": "Starting index for processing dataset samples"}
    )
    max_samples: int = field(
        default=-1,
        metadata={"help": "Maximum number of samples to process. -1 means process all samples"}
    )

def load_safetensors_weights(model_path):
    """Load weights from single or multiple safetensors files."""
    # Try single file first (like AutoMoT 2B)
    single_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_file):
        print(f"Loading from single file: {single_file}")
        return load_file(single_file)
    
    # Try multiple files (like Qwen3VL-4B)
    pattern = os.path.join(model_path, "model-*.safetensors")
    safetensor_files = sorted(glob.glob(pattern))
    
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")
    
    print(f"Loading from multiple files: {safetensor_files}")
    combined_state_dict = {}
    
    for file_path in safetensor_files:
        file_state_dict = load_file(file_path)
        combined_state_dict.update(file_state_dict)
        print(f"Loaded {len(file_state_dict)} parameters from {os.path.basename(file_path)}")
    
    print(f"Total loaded parameters: {len(combined_state_dict)}")
    return combined_state_dict

def convert_model_dtype_with_exceptions(model, target_dtype, exclude_buffer_patterns=None):
    if exclude_buffer_patterns is None:
        exclude_buffer_patterns = []
    
    for name, param in model.named_parameters():
        param.data = param.data.to(target_dtype)

    for name, buffer in model.named_buffers():
        should_exclude = any(pattern in name for pattern in exclude_buffer_patterns)
        
        if should_exclude:
            print(f"⊗ Skipped buffer: {name} (kept as {buffer.dtype})")
        else:
            buffer.data = buffer.data.to(target_dtype)
            print(f"✓ Converted buffer: {name} to {target_dtype}")   
    return model

def load_model_mot(device):
    parser = HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses(args=[])

    assert torch.cuda.is_available(), "CUDA is required"
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load configs from Qwen3VL model
    qwen3vl_config_path = model_args.qwen3vl_path
    
    # Load the unified config and extract text_config and vision_config
    with open(f"{qwen3vl_config_path}/config.json", "r") as f:
        full_config = json.load(f)
    
    # Extract and create LLM config using Qwen3VL LLM with mRoPE support
    text_config_dict = full_config["text_config"]
    llm_config = Qwen3VLTextConfig(**text_config_dict)
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = True #False
    llm_config.layer_module = "Qwen3VLMoTDecoderLayer"  # Disable MoT for debugging
    # llm_config.layer_module = "Qwen3VLMoTDecoderLayer"  
    llm_config.mot_num_attention_heads = model_args.mot_num_attention_heads
    llm_config.mot_num_key_value_heads = model_args.mot_num_key_value_heads
    llm_config.mot_intermediate_size = model_args.mot_intermediate_size

    # Extract and create Vision config
    vision_config_dict = full_config["vision_config"]
    vit_config = Qwen3VLVisionConfig(**vision_config_dict)

    config = AutoMoTConfig(
        visual_gen=inference_args.visual_gen,
        visual_und=inference_args.visual_und,
        llm_config=llm_config,
        vision_config=vit_config,  # Changed from vit_config to vision_config
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        connector_act=model_args.connector_act,
        interpolate_pos=False,
        reasoning_query_dim=model_args.reasoning_query_dim,
        reasoning_query_tokens=model_args.reasoning_query_max_num_tokens,
        action_query_dim=model_args.action_query_dim,
        action_query_tokens=model_args.action_query_tokens,
    )

    # Initialize model with Qwen3VL LLM (supports mRoPE)
    language_model = Qwen3VLForConditionalGenerationMoT(llm_config)
    vit_model = Qwen3VLVisionModel(vit_config)
        
    model = AutoMoT(language_model, vit_model, config)

    device_map = {"": "cuda:0"}

    # Load converted AutoMoT checkpoint manually (accelerate has weight issues)
    print(f"Loading converted AutoMoT checkpoint from {model_args.model_path}...")
    
    state_dict = load_safetensors_weights(model_args.model_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Filter out lm_head.weight from missing keys if tie_word_embeddings=True
    actual_missing_keys = [k for k in missing_keys if k != 'language_model.lm_head.weight']
    print(f"Loaded weights: {len(actual_missing_keys)} missing, {len(unexpected_keys)} unexpected")
    
    if actual_missing_keys:
        print(f"Missing keys: {actual_missing_keys[:10]}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}")
        
    # Move to device and siet dtype
    model = model.to(device_map[""]).eval()
    model = convert_model_dtype_with_exceptions(
        model,
        torch.bfloat16,
        exclude_buffer_patterns=['inv_freq']
    )
    
    # Verify tie_word_embeddings is working correctly
    embed_weight = model.language_model.model.embed_tokens.weight
    lm_head_weight = model.language_model.lm_head.weight
    weights_tied = embed_weight is lm_head_weight
    weights_equal = torch.equal(embed_weight, lm_head_weight)
    embed_norm = torch.norm(embed_weight).item()
    
    if not weights_tied or not weights_equal:
        print("WARNING: tie_word_embeddings may not be working correctly!")
    elif embed_norm < 10:
        print("WARNING: embed_tokens weights appear to be randomly initialized!")
    else:
        print("✓ tie_word_embeddings is working correctly")
    
    # Explicitly move any remaining CPU tensors to GPU
    print("Ensuring all parameters are on CUDA...")
    for name, param in model.named_parameters():
        if param.device.type == 'cpu':
            print(f"Moving {name} from CPU to CUDA")
            param.data = param.data.to("cuda:0")
    
    for name, buffer in model.named_buffers():
        if buffer.device.type == 'cpu':
            print(f"Moving buffer {name} from CPU to CUDA")
            buffer.data = buffer.data.to("cuda:0")

    print("Model loaded successfully")


    return model

def attach_debugger():
    import debugpy
    debugpy.listen(5683)
    print("Waiting for debugger!")
    debugpy.wait_for_client()
    print("Attached!")

def build_cleaned_prompt_and_modes(target_point_speed):

    if isinstance(target_point_speed, torch.Tensor):
        tp = target_point_speed.detach().cpu().view(-1)
        speed, x, y = float(tp[0].item()), float(tp[1].item()), float(tp[2].item())
    elif isinstance(target_point_speed, np.ndarray):
        tp = target_point_speed.reshape(-1)
        speed, x, y = float(tp[0]), float(tp[1]), float(tp[2])
    elif isinstance(target_point_speed, (list, tuple)):
        assert len(target_point_speed) >= 3
        speed, x, y = float(target_point_speed[0]), float(target_point_speed[1]), float(target_point_speed[2])
    else:
        raise TypeError(f"Unsupported type for target_point: {type(target_point_speed)}")

    x_str = f"{x:.6f}"
    y_str = f"{y:.6f}"
    prompt = f"Your target point is ({x_str}, {y_str}), and your current velocity is {speed:.2f} m/s. Predict the driving actions ( now, +1s, +2s) and plan the trajectory for the next 3 seconds."

    understanding_output = False
    reasoning_output = True

    return prompt, understanding_output, reasoning_output

def parse_decision_sequence(decision_str):
    """
    Parse decision sequence from model output.
    
    Args:
        decision_str: String like '<|im_start|> stop, accelerate, stop<|im_end|>'
                      or 'stop, accelerate, stop'
    
    Returns:
        tuple: (decision_now, decision_1s, decision_2s) - three decisions as strings
               Returns (None, None, None) if parsing fails
    
    Example:
        >>> parse_decision_sequence('<|im_start|> stop, accelerate, stop<|im_end|>')
        ('stop', 'accelerate', 'stop')
    """
    if not decision_str or not isinstance(decision_str, str):
        return (None, None, None)
    
    # Remove special tokens
    cleaned = decision_str.replace('<|im_start|>', '').replace('<|im_end|>', '')
    # Strip whitespace
    cleaned = cleaned.strip()
    
    # Split by comma
    parts = [p.strip() for p in cleaned.split(',')]
    
    # Ensure we have exactly 3 decisions
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    elif len(parts) == 2:
        return (parts[0], parts[1], None)
    elif len(parts) == 1:
        return (parts[0], None, None)
    else:
        return (None, None, None)

def split_prompt(prompt_cleaned):
    """
    Split the prompt into two sentences.
    
    Args:
        prompt_cleaned: String like 'Your target point is (53.101654, 0.201010), and your current velocity is 0.00 m/s. Predict the driving actions ( now, +1s, +2s) and plan the trajectory for the next 3 seconds.'
    
    Returns:
        tuple: (sentence1, sentence2)
    
    Example:
        >>> split_prompt('Your target point is (...). Predict the driving actions...')
        ('Your target point is (...).', 'Predict the driving actions...')
    """
    if not prompt_cleaned or not isinstance(prompt_cleaned, str):
        return (None, None)
    
    # Find the first period followed by space (end of first sentence)
    # Pattern: "...m/s. Predict..."
    split_marker = "m/s. "
    if split_marker in prompt_cleaned:
        idx = prompt_cleaned.find(split_marker)
        sentence1 = prompt_cleaned[:idx + 4]  # Include "m/s."
        sentence2 = prompt_cleaned[idx + 5:]  # Skip "m/s. "
        return (sentence1.strip(), sentence2.strip())
    
    # Fallback: split by ". " if marker not found
    if ". " in prompt_cleaned:
        idx = prompt_cleaned.find(". ")
        sentence1 = prompt_cleaned[:idx + 1]
        sentence2 = prompt_cleaned[idx + 2:]
        return (sentence1.strip(), sentence2.strip())
    
    return (prompt_cleaned, None)

class MOTAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		if IS_BENCH2DRIVE:
			self.save_name = path_to_conf_file.split('+')[-1]
			self.config_path = path_to_conf_file.split('+')[0]
		else:
			now = datetime.datetime.now()
			self.config_path = path_to_conf_file
			self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		# Load diffusion policy
		print("Loading diffusion policy...")
		self.config = create_carla_config()
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		checkpoint_base_path = self.config.get('training', {}).get('checkpoint_dir', "/home/wang/Project/MoT-DP/checkpoints/carla_dit_best")
		checkpoint_path = os.path.join(checkpoint_base_path, "carla_policy_best.pt")
		self.net = load_best_model(checkpoint_path, self.config, device)
		self.net = self.net.to(torch.bfloat16)
		if hasattr(self.net, 'obs_encoder'):
			self.net.obs_encoder = self.net.obs_encoder.to(torch.bfloat16)
		print("✓ Diffusion policy loaded (bfloat16).")
		
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			import gc
			gc.collect()


		# Load MoT model
		print("Loading MoT model...")
		parser = HfArgumentParser((ModelArguments, InferenceArguments))
		model_args, inference_args = parser.parse_args_into_dataclasses(args=[])
		self.inference_args = inference_args  
		self.AutoMoT = load_model_mot(device)
		tokenizer = AutoTokenizer.from_pretrained(model_args.qwen3vl_path)
		tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
		self.AutoMoT.language_model.tokenizer = tokenizer
		self.inferencer = InterleaveInferencer(
        model=self.AutoMoT,
        vae_model=None,
        tokenizer=tokenizer,
        vae_transform=None,
        vit_transform=None,  # Not used for Qwen3VL, handled internally by model
        new_token_ids=new_token_ids,
        max_num_tokens=inference_args.max_num_tokens,
        visual_gen=True,  # Enable visual generation to initialize query tokens
        visual_und=True,  # Enable visual understanding
    	)
		print("✓ MoT model loaded.")

		self.turn_controller = LateralPIDController(inference_mode=True)  # Set inference_mode=True for model predictions
		self.speed_controller = t_u.PIDController(k_p=1.35, k_i=1.0, k_d=2.0, n=20)  # kp 1.75
		
		# Control config 
		self.carla_fps = 20
		self.wp_dilation = 1
		self.data_save_freq = 5
		self.brake_speed = 0.4
		self.brake_ratio = 1.1
		self.clip_delta = 1.0
		self.clip_throttle = 1.0
		self.stuck_threshold = 200 #800
		self.stuck_helper_threshold = 100
		self.stuck_creeper_threshold = 100

		self.creep_duration = 15
		self.creep_throttle = 0.4
		
		# Stuck detection
		self.stuck_detector = 0
		self.stuck_helper = 0
		self.stuck_creeper = 0
		self.force_move = 0

		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = 0
		
		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0
		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
		self.lat_ref, self.lon_ref = 42.0, 2.0
		control = carla.VehicleControl()
		control.steer = 0.0
		control.throttle = 0.0
		control.brake = 0.0	
		self.prev_control = control
		self.control = control  # Store control for UKF prediction
		
		# Initialize Unscented Kalman Filter 
		self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate
		if USE_UKF:
			self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
			self.ukf = UKF(dim_x=4,
						   dim_z=4,
						   fx=bicycle_model_forward,
						   hx=measurement_function_hx,
						   dt=self.carla_frame_rate,
						   points=self.points,
						   x_mean_fn=state_mean,
						   z_mean_fn=measurement_mean,
						   residual_x=residual_state_x,
						   residual_z=residual_measurement_h)
			# State noise, same as measurement because we initialize with the first measurement later
			self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
			# Measurement noise
			self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
			self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
			# Used to set the filter state equal the first measurement
			self.filter_initialized = False
			# Stores the last filtered positions of the ego vehicle
			self.state_log = deque(maxlen=20)

		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = self.save_name

		self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
		self.save_path.mkdir(parents=True, exist_ok=False)

		(self.save_path / 'rgb_front').mkdir()
		(self.save_path / 'meta').mkdir()
		(self.save_path / 'bev').mkdir()
		(self.save_path / 'lidar_bev').mkdir()
		
		# Initialize lidar buffer for combining two frames
		self.lidar_buffer = deque(maxlen=2)
		self.lidar_step_counter = 0
		self.last_ego_transform = None
		self.last_lidar = None
		
		obs_horizon = self.config.get('obs_horizon', 4)   
		self.obs_horizon = obs_horizon
		self.lidar_bev_history = deque(maxlen=obs_horizon*10) 
		self.rgb_history = deque(maxlen=obs_horizon*10)
		self.speed_history = deque(maxlen=obs_horizon*10)
		self.theta_history = deque(maxlen=obs_horizon*10)		# tg = tick_data['target_point']
		# tg[1] = 0.05
		# target_point = torch.from_numpy(tg).unsqueeze(0).float().to('cuda', dtype=torch.float32)e(maxlen=obs_horizon*10)
		self.throttle_history = deque(maxlen=obs_horizon*10)
		self.next_command_history = deque(maxlen=obs_horizon*10)
		self.target_point_history = deque(maxlen=obs_horizon*10)
		self.waypoint_history = deque(maxlen=obs_horizon*10)
		self.throttle_history = deque(maxlen=obs_horizon*10) 
		self.brake_history = deque(maxlen=obs_horizon*10) 
		self.obs_accumulate_counter = 0
		
		# Store predicted trajectory for BEV visualization
		self.last_pred_traj = None  # Store the last predicted trajectory (in ego frame)
		self.last_dp_pred_traj = None  # Store the last DP refined trajectory (in ego frame)
		self.last_target_point = None  # Store the last target point (in ego frame)

	def _init(self):
		# Use _global_plan_world_coord directly (already in CARLA coordinates)
		# This avoids the GPS-to-CARLA conversion which can fail when fsolve doesn't converge
		# Get lat_ref/lon_ref from CARLA map directly
		try:
			world_map = CarlaDataProvider.get_map()
			xodr = world_map.to_opendrive()
			tree = ET.ElementTree(ET.fromstring(xodr))
			
			# Default values if not found in OpenDRIVE
			self.lat_ref = 42.0
			self.lon_ref = 2.0
			
			for opendrive in tree.iter('OpenDRIVE'):
				for header in opendrive.iter('header'):
					for georef in header.iter('geoReference'):
						if georef.text:
							str_list = georef.text.split(' ')
							for item in str_list:
								if '+lat_0' in item:
									self.lat_ref = float(item.split('=')[1])
								if '+lon_0' in item:
									self.lon_ref = float(item.split('=')[1])
		except Exception as e:
			# Fallback: try fsolve (might not converge)
			try:
				locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
				lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
				earth_radius_equa = 6378137.0
				def equations(variables):
					x, y = variables
					eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa)
								 - math.cos(x * math.pi / 180.0) * y)
					eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius_equa
								 * math.cos(x * math.pi / 180.0) + locy - math.cos(x * math.pi / 180.0) * earth_radius_equa
								 * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
					return [eq1, eq2]
				initial_guess = [0.0, 0.0]
				solution = fsolve(equations, initial_guess)
				self.lat_ref, self.lon_ref = solution[0], solution[1]
			except Exception as e2:
				self.lat_ref, self.lon_ref = 0.0, 0.0
		

		self.route_planner_min_distance = 7.5
		self.route_planner_max_distance = 50.0
		self._route_planner = RoutePlanner(self.route_planner_min_distance, self.route_planner_max_distance,
										   self.lat_ref, self.lon_ref)
		
		if len(self._global_plan_world_coord) > 0:
			first_wp = self._global_plan_world_coord[0]
		
		# Use _global_plan_world_coord with gps=False (recommended, GPS is deprecated in nav_planner.py)
		self._route_planner.set_route(self._global_plan_world_coord, gps=False)
		
				
		# Initialize command tracking 
		self.commands = deque(maxlen=2)
		self.commands.append(4)
		self.commands.append(4)
		self.target_point_prev = [1e5, 1e5, 1e5]
		self.last_command = -1
		self.last_command_tmp = -1
		
		self.initialized = True
		self.metric_info = {}
		self._hic = DisplayInterface()

	def _build_obs_dict(self, tick_data, lidar, rgb_front, speed, theta, target_point, cmd_one_hot, waypoint):
		"""
		Build observation dictionaries from historical data.
		
		Returns:
			lidar_stacked: (1, obs_horizon, C, H, W) stacked lidar BEV images
			ego_status_stacked: (1, obs_horizon, 14) concatenated ego status features
			rgb_stacked: (1, 5, C, H, W) stacked RGB images
		"""
		# Build obs_dict from historical observations
		lidar_history_list = list(self.lidar_bev_history)
		speed_history_list = list(self.speed_history)
		target_point_history_list = list(self.target_point_history)
		cmd_history_list = list(self.next_command_history)
		theta_history_list = list(self.theta_history)
		throttle_history_list = list(self.throttle_history)
		brake_history_list = list(self.brake_history)
		rgb_history_list = list(self.rgb_history)
		waypoint_history_list = list(self.waypoint_history)
		
		lidar_list = [lidar_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(lidar_history_list)]
		speed_list = [speed_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(speed_history_list)]
		target_point_list = [target_point_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(target_point_history_list)]
		cmd_list = [cmd_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(cmd_history_list)]
		theta_list = [theta_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(theta_history_list)]
		throttle_list = [throttle_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(throttle_history_list)]
		brake_list = [brake_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(brake_history_list)]
		waypoint_list = [waypoint_history_list[-1 - i*10] for i in range(self.obs_horizon) if -1 - i*10 >= -len(waypoint_history_list)]
		
		# Reverse to get chronological order (oldest to newest)
		lidar_list = lidar_list[::-1]
		speed_list = speed_list[::-1]
		target_point_list = target_point_list[::-1]
		cmd_list = cmd_list[::-1]
		theta_list = theta_list[::-1]
		throttle_list = throttle_list[::-1]
		brake_list = brake_list[::-1]
		waypoint_list = waypoint_list[::-1]
		
		# sample every 5 for rgb from the end (t0, t-5, t-10, t-15)
		rgb_list = [rgb_history_list[-1 - i*5] for i in range(4) if -1 - i*5 >= -len(rgb_history_list)]
		rgb_list = rgb_list[::-1]  
		
		# Stack along time dimension
		lidar_stacked = torch.stack(lidar_list, dim=0).unsqueeze(0)  # (1, obs_horizon, C, H, W)
		speed_stacked = torch.cat(speed_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		theta_stacked = torch.cat(theta_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		throttle_stacked = torch.cat(throttle_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		brake_stacked = torch.cat(brake_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 1)
		cmd_stacked = torch.cat(cmd_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 6)
		target_point_stacked = torch.cat(target_point_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		waypoint_stacked = torch.stack(waypoint_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		rgb_stacked = torch.stack(rgb_list, dim=0).unsqueeze(0)  # (1, 5, C, H, W)
		

		current_pos = tick_data['gps']  
		current_theta = tick_data['theta']  # This is already preprocessed: compass - 90°
		
		# Build rotation matrix for world-to-ego transformation
		# R = [[cos, -sin], [sin, cos]] is the ego-to-world rotation
		# R.T = [[cos, sin], [-sin, cos]] is the world-to-ego rotation
		# This matches inverse_conversion_2d: R.T @ (point - translation)
		cos_theta = np.cos(current_theta)
		sin_theta = np.sin(current_theta)
		current_R = np.array([
			[cos_theta, sin_theta],
			[-sin_theta, cos_theta]
		])  # This is R.T, for world-to-ego transformation
		
		# Transform each historical waypoint to current frame
		waypoint_relative_list = []
		for i in range(self.obs_horizon):
			past_waypoint = waypoint_stacked[0, i].cpu().numpy()  # [x, y] in global coordinates
			# Transform from world to ego frame: R.T @ (past - current)
			relative_waypoint = current_R @ (past_waypoint - current_pos)
			waypoint_relative_list.append(torch.from_numpy(relative_waypoint).float().to('cuda'))
		
		waypoint_relative_stacked = torch.stack(waypoint_relative_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		
		target_point_transformed_list = []
		for i in range(self.obs_horizon):
			past_world_pos = waypoint_list[i].cpu().numpy()  # [x, y] in world coordinates
			past_theta = theta_list[i].squeeze().cpu().item()  # theta value for past frame (already preprocessed)
			target_in_past_ego = target_point_list[i].squeeze().cpu().numpy()  # [x, y] in past ego frame
			
			# Build past ego-to-world rotation matrix
			cos_past = np.cos(past_theta)
			sin_past = np.sin(past_theta)
			past_R_ego_to_world = np.array([
				[cos_past, -sin_past],
				[sin_past, cos_past]
			])  # This is R, for ego-to-world transformation
			
			# Step 1: Transform target point from past ego frame to world frame
			target_world = past_R_ego_to_world @ target_in_past_ego + past_world_pos
			
			# Step 2: Transform from world frame to current ego frame
			target_in_current_ego = current_R @ (target_world - current_pos)
			
			target_point_transformed_list.append(torch.from_numpy(target_in_current_ego).float().to('cuda'))
		
		target_point_transformed_stacked = torch.stack(target_point_transformed_list, dim=0).unsqueeze(0)  # (1, obs_horizon, 2)
		
		
		# Concatenate all ego status features: speed + theta + throttle + brake + cmd + target_point + waypoint_relative
		# ego_status_stacked: (1, obs_horizon, 1+1+1+1+6+2+2) = (1, obs_horizon, 14)
		ego_status_stacked = torch.cat([
			speed_stacked,                        # (1, obs_horizon, 1)
			theta_stacked,                        # (1, obs_horizon, 1)
			cmd_stacked,                          # (1, obs_horizon, 6)
			target_point_transformed_stacked,     # (1, obs_horizon, 2) - target points transformed to current ego frame
			waypoint_relative_stacked             # (1, obs_horizon, 2) - ego positions in current frame
		], dim=-1)  # Concatenate along feature dimension
		
		return lidar_stacked, ego_status_stacked, rgb_stacked


	def sensors(self):
		sensors =  [
				{
					'type': 'sensor.camera.rgb',
					'x': -1.50, 'y': 0.0, 'z': 2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 1024, 'height': 512, 'fov': 110,
					'id': 'CAM_FRONT'
					},
				# lidar
				{
          			'type': 'sensor.lidar.ray_cast',
          			'x': 0.0, 'y': 0.0, 'z': 2.5,
          			'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
          			'id': 'LIDAR'
      				},
				# imu
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'IMU'
					},
				# gps
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'GPS'
					},
				# speed
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'SPEED'
					},
				]
		
		if IS_BENCH2DRIVE:
			sensors += [
					{	
						'type': 'sensor.camera.rgb',
						'x': 0.0, 'y': 0.0, 'z': 50.0,
						'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
						'width': 512, 'height': 512, 'fov': 5 * 10.0,
						'id': 'bev'
					}]
		return sensors

	def tick(self, input_data):
		self.step += 1
		rgb_front = cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		lidar_ego = lidar_to_ego_coordinate(input_data['LIDAR'])
		
		gps_full = input_data['GPS'][1]  # [lat, lon, altitude]
		gps_pos = self._route_planner.convert_gps_to_carla(gps_full)
		
		# Handle compass NaN 
		compass_raw = input_data['IMU'][1][-1]
		if math.isnan(compass_raw):
			print("compass sends nan!!!")
			compass_raw = 0.0
		
		# Preprocess compass to CARLA coordinate system
		compass = t_u.preprocess_compass(compass_raw)
		
		# Get speed for UKF
		speed = input_data['SPEED'][1]['speed']
		
		# Apply Unscented Kalman Filter 
		if USE_UKF:
			if not self.filter_initialized:
				self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
				self.filter_initialized = True

			self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
			self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
			filtered_state = self.ukf.x

			self.state_log.append(filtered_state)
			gps_filtered = filtered_state[0:2]
			compass_filtered = filtered_state[2]
		else:
			gps_filtered = np.array([gps_pos[0], gps_pos[1]])
			compass_filtered = compass
		
		# Combine two frames of lidar data using algin_lidar
		# Use filtered GPS for lidar alignment
		if self.last_lidar is not None and self.last_ego_transform is not None:
			# Calculate relative transformation between current and last frame
			current_pos = np.array([gps_filtered[0], gps_filtered[1], 0.0])
			last_pos = np.array([self.last_ego_transform['gps'][0], self.last_ego_transform['gps'][1], 0.0])
			relative_translation = current_pos - last_pos
			
			# Calculate relative rotation using filtered compass
			current_yaw = compass_filtered
			last_yaw = self.last_ego_transform['compass']
			relative_rotation = current_yaw - last_yaw
			
			# Rotate difference vector from global to local coordinate system
			rotation_matrix = np.array([[np.cos(current_yaw), -np.sin(current_yaw), 0.0],
										[np.sin(current_yaw), np.cos(current_yaw), 0.0], 
										[0.0, 0.0, 1.0]])
			relative_translation_local = rotation_matrix.T @ relative_translation
			
			# Align the last lidar to current coordinate system
			lidar_last = algin_lidar(self.last_lidar, relative_translation_local, relative_rotation)
			# Combine lidar frames
			lidar_combined = np.concatenate((lidar_ego, lidar_last), axis=0)
		else:
			lidar_combined = lidar_ego
		
		# Store current frame for next iteration (use filtered values)
		self.last_lidar = lidar_ego
		self.last_ego_transform = {'gps': gps_filtered, 'compass': compass_filtered}
		
		# Generate lidar BEV image from combined lidar data
		lidar_bev_img = generate_lidar_bev_images(
			np.copy(lidar_combined), 
			saving_name=None, 
			img_height=448, 
			img_width=448
		)
		# Convert BEV image to tensor format for interfuser_bev_encoder backbone
		lidar_bev_tensor = torch.from_numpy(lidar_bev_img).permute(2, 0, 1).float() / 255.0
		
		# Process other sensors
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		
		result = {
				'rgb_front': rgb_front,
				'lidar_bev': lidar_bev_tensor,
				'gps': gps_filtered,  # Use UKF filtered CARLA coordinates
				'speed': speed,
				'compass': compass_filtered,  # Use UKF filtered compass
				'bev': bev
				}
		
		waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))
		

		
		# if len(waypoint_route) > 2:
		# 	target_point, far_command = waypoint_route[1]
		# 	next_target_point, next_far_command = waypoint_route[2]
		# elif len(waypoint_route) > 1:
		# 	target_point, far_command = waypoint_route[1]
		# 	next_target_point, next_far_command = waypoint_route[1]
		# else:
		# 	target_point, far_command = waypoint_route[0]
		# 	next_target_point, next_far_command = waypoint_route[0]

		if len(waypoint_route) > 2:
			target_point, far_command = waypoint_route[-1]
			next_target_point, next_far_command = waypoint_route[-1]
		elif len(waypoint_route) > 1:
			target_point, far_command = waypoint_route[-1]
			next_target_point, next_far_command = waypoint_route[-1]
		else:
			target_point, far_command = waypoint_route[-1]
			next_target_point, next_far_command = waypoint_route[-1]


		if self.last_command_tmp != far_command:
			self.last_command = self.last_command_tmp
		self.last_command_tmp = far_command
		
		if hasattr(target_point, '__iter__') and len(target_point) >= 2:
			if (target_point[:2] != self.target_point_prev[:2]).any() if isinstance(target_point, np.ndarray) else (list(target_point[:2]) != list(self.target_point_prev[:2])):
				self.target_point_prev = target_point
				self.commands.append(far_command.value)
		
		result['next_command'] = self.commands[-2]
		ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass']) #result['compass'])
		ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], result['gps'], result['compass']) #result['compass'])
		
		result['target_point'] = ego_target_point  # numpy array (2,)
		result['next_target_point'] = ego_next_target_point  # numpy array (2,)
		result['theta'] = compass_filtered  # Use UKF filtered compass


		# ==============debug=========		
		print(f"  target_point (world): {target_point[:2]}")
		print(f"  ego position (gps): {result['gps']}")
		print(f"  compass (heading): {result['compass']:.4f} rad ({np.rad2deg(result['compass']):.2f} deg)")
		print(f"  ego_target_point: {ego_target_point}")

		return result
	
	def control_pid(self, route_waypoints, velocity, speed_waypoints):
		"""
		Predicts vehicle control with a PID controller.
		
		Args:
			route_waypoints: (1, N, 2) tensor in ego frame [x_forward, y_left]
			velocity: float, current speed in m/s
			speed_waypoints: (1, N, 2) tensor for speed calculation
		"""
		assert route_waypoints.size(0) == 1
		route_waypoints_np = route_waypoints[0].data.cpu().numpy()  # (N, 2)
		speed = velocity  # Already a float
		speed_waypoints_np = speed_waypoints[0].data.cpu().numpy()  # (N, 2)
		
		# MoT trajectory: 6 points, 0.5s interval each, total 3s
		# Point indices: 0(0.5s), 1(1.0s), 2(1.5s), 3(2.0s), 4(2.5s), 5(3.0s)
		mot_waypoint_interval = 0.5  # seconds between waypoints
		one_second_idx = 2  # point[1] is at 1.0s
		half_second_idx = 0  # point[0] is at 0.5s
		
		if speed_waypoints_np.shape[0] >= 2:
			# Displacement from 0.5s to 1.0s position, multiply by 2 to get m/s
			# desired_speed = (np.linalg.norm(speed_waypoints_np[one_second_idx] - speed_waypoints_np[half_second_idx]) * 2.0)
			desired_speed = np.linalg.norm(speed_waypoints_np[one_second_idx] - speed_waypoints_np[half_second_idx])
			# desired_speed += 0.75*(np.linalg.norm(speed_waypoints_np[0]) * 2.0)
		else:
			# Fallback: use first point distance, assuming it represents 0.5s travel
			desired_speed = np.linalg.norm(speed_waypoints_np[0]) * 2.0

		brake = ((desired_speed < self.brake_speed) or ((speed / max(desired_speed, 1e-5)) > self.brake_ratio))
		
		delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.clip_throttle)
		throttle = throttle if not brake else 0.0

		# Speed limit 
		# max_speed = 6.0  # m/s
		# if speed > max_speed:
		# 	print(f"⚠️ Speed limit exceeded: {speed:.2f} m/s > {max_speed:.2f} m/s, applying brake")
		# 	brake = True
		# 	throttle = 0.1
		# else:
		# 	# Clamp desired speed to max speed
		# 	desired_speed = min(desired_speed, max_speed)
			
		# 	brake = ((desired_speed < self.brake_speed) or ((speed / max(desired_speed, 1e-5)) > self.brake_ratio))
			
		# 	delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
		# 	throttle = self.speed_controller.step(delta)
		# 	throttle = np.clip(throttle, 0.0, self.clip_throttle)
		# 	throttle = throttle if not brake else 0.0
		

		route_interp = self.interpolate_waypoints(route_waypoints_np)
		
		# DEBUG: Print interpolated route and lookahead point
		speed_kmh = speed * 3.6
		n_lookahead = np.clip(0.9755 * speed_kmh + 1.9153, 24, 105) / 10 - 2
		n_lookahead = int(min(n_lookahead, route_interp.shape[0] - 1))
		lookahead_pt = route_interp[n_lookahead]
		yaw_rad = np.arctan2(lookahead_pt[1], lookahead_pt[0])
		yaw_deg = np.degrees(yaw_rad)
		print(f"[DEBUG] control_pid: speed={speed:.2f}m/s, n_lookahead={n_lookahead}")
		print(f"[DEBUG] lookahead_pt: x={lookahead_pt[0]:.3f}, y={lookahead_pt[1]:.3f}, yaw_deg={yaw_deg:.2f}")
		
		steer = self.turn_controller.step(route_interp, speed)
		print(f"[DEBUG] steer output: {steer:.4f}")
		steer = np.clip(steer, -1.0, 1.0)
		steer = round(steer, 3)
		
		
		return steer, throttle, brake
	
	def interpolate_waypoints(self, waypoints):
		"""
		Interpolate waypoints to be 0.1m apart
		
		Args:
			waypoints: (N, 2) numpy array in ego frame [x_forward, y_left]
			
		Returns:
			interp_points: (M, 2) numpy array with points 0.1m apart
		"""
		waypoints = waypoints.copy()
		# Add origin point at the beginning
		waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
		shift = np.roll(waypoints, 1, axis=0)
		shift[0] = shift[1]
		
		dists = np.linalg.norm(waypoints - shift, axis=1)
		dists = np.cumsum(dists)
		dists += np.arange(0, len(dists)) * 1e-4  # Prevents dists not being strictly increasing
		
		interp = PchipInterpolator(dists, waypoints, axis=0)
		
		x = np.arange(0.1, dists[-1], 0.1)
		
		interp_points = interp(x)
		
		if interp_points.shape[0] == 0:
			interp_points = waypoints[None, -1]
		
		return interp_points
	
	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)

		# Prepare current observations
		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
		theta = torch.FloatTensor([float(tick_data['theta'])]).view(1,1).to('cuda', dtype=torch.float32)
		lidar = tick_data['lidar_bev'].to('cuda', dtype=torch.float32)
		
		rgb_front = torch.from_numpy(tick_data['rgb_front']).permute(2, 0, 1).float() / 255.0
		rgb_front = rgb_front.to('cuda', dtype=torch.float32)
		waypoint = torch.from_numpy(tick_data['gps']).float().to('cuda', dtype=torch.float32)
		target_point = torch.from_numpy(tick_data['target_point']).unsqueeze(0).float().to('cuda', dtype=torch.float32)
		next_target_point = torch.from_numpy(tick_data['next_target_point']).unsqueeze(0).float().to('cuda', dtype=torch.float32)
		# tg = tick_data['target_point']
		# tg[1] = 0.05
		# target_point = torch.from_numpy(tg).unsqueeze(0).float().to('cuda', dtype=torch.float32)

		# Accumulate observation history into buffers 
		self.lidar_bev_history.append(lidar)
		self.rgb_history.append(rgb_front)
		self.speed_history.append(speed)
		self.target_point_history.append(target_point)
		self.next_command_history.append(cmd_one_hot)
		self.theta_history.append(theta)
		self.waypoint_history.append(waypoint)
		
		# Append throttle and brake from previous step (or 0 for first step)
		if self.step < 1:
			# First step: initialize with 0
			self.throttle_history.append(torch.tensor(0.0).view(1, 1).to('cuda'))
			self.brake_history.append(torch.tensor(0.0).view(1, 1).to('cuda'))
		else:
			# Use control from previous step
			prev_control = self.prev_control if self.prev_control is not None else carla.VehicleControl()
			self.throttle_history.append(torch.tensor(prev_control.throttle).view(1, 1).to('cuda'))
			self.brake_history.append(torch.tensor(prev_control.brake).view(1, 1).to('cuda'))
		
		# Buffer size = 31 frames (for obs_horizon with 10x sampling)
		BUFFER_PHASE = 31     # Fill buffer to minimum required size
		
		if self.step < BUFFER_PHASE:
			# Warmup phase: use previous control or default
			control = self.prev_control
			self.pid_metadata = {}
			self.pid_metadata['agent'] = 'warmup_phase'
			self.pid_metadata['step'] = self.step
		else:
			# Build observation dict
			lidar_stacked, ego_status_stacked, rgb_stacked = self._build_obs_dict(
				tick_data, lidar, rgb_front, speed, theta, target_point, 
				cmd_one_hot, waypoint
			)
			
			rgb_pil_list = []
			for i in range(rgb_stacked.shape[1]): 
				rgb_tensor = rgb_stacked[0, i]  # (C, H, W)
				rgb_np = (rgb_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
				rgb_pil = Image.fromarray(rgb_np, mode='RGB')
				rgb_pil_list.append(rgb_pil)
			
			# lidar_stacked shape: (1, obs_horizon, C, H, W)
			lidar_tensor = lidar_stacked[0, -1]  # (C, H, W) - last frame
			lidar_np = (lidar_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
			lidar_pil = Image.fromarray(lidar_np, mode='RGB')
			lidar_pil_list = [lidar_pil]  
			
			if self.stuck_helper > 0:
				target_point_speed=torch.cat([speed, next_target_point], dim=-1)
				print("Get stucked! Trigger the stuck helper!")
			else:
				target_point_speed=torch.cat([speed, target_point], dim=-1)  # (1, 3)

			prompt_cleaned, understanding_output, reasoning_output = build_cleaned_prompt_and_modes(target_point_speed)

			predicted_answer = self.inferencer(
				image=rgb_pil_list,  
				front=[rgb_pil_list[-1]],  
				lidar=lidar_pil_list,
				v_target_point=target_point_speed,
				text=prompt_cleaned,
				understanding_output=understanding_output,
				reasoning_output=reasoning_output,
				max_think_token_n=self.inference_args.max_num_tokens,
				do_sample=False,
				text_temperature=0.0,
			)

			pred_traj = predicted_answer['traj']  # Shape: (1, 6, 2) in ego frame [x_forward, y_left]
			pred_decision = predicted_answer['text']
			
			self.last_pred_traj = pred_traj.squeeze(0).float().cpu().numpy()  # (6, 2) in [x, y] format
			self.last_target_point = target_point.squeeze(0).float().cpu().numpy()  # (2,) in [x, y] format
		
			# DP trajectory refinement
			# Add batch dimension to features: (seq_len, feat_dim) -> (1, seq_len, feat_dim)
			dp_vit_feat = predicted_answer['dp_vit_feat']
			if dp_vit_feat.dim() == 2:
				dp_vit_feat = dp_vit_feat.unsqueeze(0)  # (1, Nvit, C)
			
			reason_feat = predicted_answer['reason_feat']
			if reason_feat.dim() == 2:
				reason_feat = reason_feat.unsqueeze(0)  # (1, Nr, C)
			
			dp_obs_dict = {
                    'lidar_bev': lidar_stacked,
                    'ego_status': ego_status_stacked,  
                    'gen_vit_tokens': dp_vit_feat,
                    'reasoning_query_tokens': reason_feat[:7],
                    'anchor': predicted_answer['traj']  # Pass anchor for truncated diffusion
                }
			dp_pred_traj = self.net.predict_action(dp_obs_dict)
			self.last_dp_pred_traj = dp_pred_traj['action'].squeeze(0).copy()  # (6, 2) in [x, y] format
			print("dp_pred_traj:", dp_pred_traj)
			# self.last_dp_pred_traj=np.zeros_like(self.last_dp_pred_traj) # debug

			# ================== control_pid method ==================
			#route_waypoints = pred_traj.float()  # (1, 6, 2) - use as route_waypoints
			#speed_waypoints = pred_traj.float()  # (1, 6, 2) - use as speed_waypoints (same for MoT)
			route_waypoints = torch.from_numpy(dp_pred_traj['action']) # (1, 6, 2) - use as route_waypoints
			speed_waypoints = torch.from_numpy(dp_pred_traj['action']) # (1, 6, 2) - use as speed_waypoints (same for MoT)
			gt_velocity = tick_data['speed']
			# import pdb; pdb.set_trace()
			# DEBUG: Print trajectory values to diagnose left-drift
			traj_np = pred_traj[0].float().cpu().numpy()  # Convert BFloat16 -> Float32 -> numpy
			print(f"[DEBUG] pred_traj (ego frame [x_fwd, y_left]):")
			for i, pt in enumerate(traj_np):
				print(f"  point[{i}]: x={pt[0]:.3f}, y={pt[1]:.3f}")
			print(f"[DEBUG] target_point: x={target_point[0,0].item():.3f}, y={target_point[0,1].item():.3f}")
			
			steer, throttle, brake = self.control_pid(route_waypoints, gt_velocity, speed_waypoints)
			
			# Restart mechanism in case the car got stuck 
			if gt_velocity < 0.1:
				if self.stuck_detector > self.stuck_threshold:
					self.stuck_helper += 1
				self.stuck_detector += 1
			
			if self.stuck_helper > self.stuck_helper_threshold or gt_velocity > 3:
					self.stuck_helper = 0
					self.stuck_detector = 0
			

			print(f"stuck_helper: {self.stuck_helper}")
			print(f"stuck_detector: {self.stuck_detector}")

			if self.stuck_helper > 0 and gt_velocity < 0.5:
				print("Stuck helper activated: hhhhh")
				throttle = 0.25
				steer = -0.15
				brake = 0.0
	

			control = carla.VehicleControl()
			control.steer = float(steer)
			control.throttle = float(throttle)
			control.brake = float(brake)
			
			# Store metadata
			self.pid_metadata = {
				'agent': 'mot',
				'steer': control.steer,
				'throttle': control.throttle,
				'brake': control.brake,
				'speed': gt_velocity,
				'command': command,
			}

			self.prev_control = control
			self.control = control  # Update control for UKF prediction in next tick
			metric_info = self.get_metric_info()
			self.metric_info[self.step] = metric_info

			if SAVE_PATH is not None:
				self.save(tick_data)

			##### Rendering ####
			ego_car_map = render_self_car(
				loc=np.array([0, 0]),
				ori=np.array([0, -1]),
				box=np.array([2.45, 1.0]),
				color=[1, 1, 0], pixels_per_meter=10, max_distance=30,
			)

			# Prepare trajectory for rendering (pred_traj - green)
			traj_for_render = pred_traj.squeeze(0).cpu().float().numpy().copy()  # (6, 2)
			traj_for_render[:, 1] = -traj_for_render[:, 1]  # Negate y: left -> right
			tp_for_render = target_point.cpu().float().numpy().copy()
			if tp_for_render.ndim == 2:
				tp_for_render = tp_for_render.squeeze(0)
			tp_for_render[1] = -tp_for_render[1]  # Negate y: left -> right
			

			trajectory = np.concatenate((traj_for_render, tp_for_render.reshape(1, 2)), axis=0)
			trajectory = trajectory[:, [1, 0]]
			trajectory[:, 0] = -trajectory[:, 0]  # y (now in col 0) 
			trajectory[:, 1] = -trajectory[:, 1]  # x (now in col 1)
			render_trajectory = render_waypoints(trajectory, pixels_per_meter=30, max_distance=20, color=(0, 255, 0))
			
			# Prepare dp_pred_traj for rendering (red)
			dp_traj_for_render = dp_pred_traj['action'].squeeze(0).copy()  # (6, 2) - already numpy
			dp_traj_for_render[:, 1] = -dp_traj_for_render[:, 1]  # Negate y: left -> right
			dp_trajectory = np.concatenate((dp_traj_for_render, tp_for_render.reshape(1, 2)), axis=0)
			dp_trajectory = dp_trajectory[:, [1, 0]]
			dp_trajectory[:, 0] = -dp_trajectory[:, 0]  # y (now in col 0) 
			dp_trajectory[:, 1] = -dp_trajectory[:, 1]  # x (now in col 1)
			render_dp_trajectory = render_waypoints(dp_trajectory, pixels_per_meter=30, max_distance=20, color=(255, 0, 0))

			ego_car_map = cv2.resize(ego_car_map, (200, 200))
			render_trajectory = cv2.resize(render_trajectory, (200, 200))
			render_dp_trajectory = cv2.resize(render_dp_trajectory, (200, 200))

			surround_map = np.clip(
				(
					ego_car_map.astype(np.float32)
					+ render_trajectory.astype(np.float32)
					+ render_dp_trajectory.astype(np.float32)
				),
				0,
				255,
			).astype(np.uint8)
			tick_data["predicted_trajectory"] = surround_map
			decision_1s, decision_2s, decision_3s = parse_decision_sequence(pred_decision)
			tick_data["decision_1s"] = decision_1s
			tick_data["decision_2s"] = decision_2s
			tick_data["decision_3s"] = decision_3s

			tick_data["rgb_raw"] = tick_data["rgb_front"]

			tick_data["rgb"] = cv2.resize(tick_data["rgb_front"], (800, 600))
			tick_data["bev_traj"] = cv2.resize(tick_data["bev_traj"], (400, 400))

			tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
				control.throttle,
				control.steer,
				control.brake,
			)
			tick_data["speed"] = "speed: %.2f Km/h, target point x: %.2f m, target point y: %.2f m" % (gt_velocity*3.6, target_point.squeeze(0).cpu().float().numpy()[0], target_point.squeeze(0).cpu().float().numpy()[1])
			
			sentence1, sentence2 = split_prompt(prompt_cleaned)
			tick_data["language_1"] = "Instruction: " + sentence1
			tick_data["language_2"] = sentence2

			tick_data["mes"] = "speed: %.2f" % gt_velocity
			tick_data["time"] = "time: %.3f" % timestamp

			surface = self._hic.run_interface(tick_data)
			tick_data["surface"] = surface

		return control

	def save(self, tick_data):
		frame = self.step 
		Image.fromarray(tick_data['rgb_front']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
		
		# Draw trajectory on BEV image if available
		bev_img = tick_data['bev'].copy()
		if self.last_pred_traj is not None:
			bev_img = self._draw_trajectory_on_bev(bev_img, self.last_pred_traj, self.last_target_point, self.last_dp_pred_traj)
		tick_data['bev_traj'] = bev_img
		Image.fromarray(bev_img).save(self.save_path / 'bev' / ('%04d.png' % frame))
		
		if 'lidar_bev' in tick_data:
			lidar_bev_tensor = tick_data['lidar_bev']
			if isinstance(lidar_bev_tensor, torch.Tensor):
				lidar_bev_tensor = lidar_bev_tensor.cpu().numpy()
			lidar_bev_img = (lidar_bev_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
			imageio.imwrite(str(self.save_path / 'lidar_bev' / (f'{frame:04d}.png')), lidar_bev_img)

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

		# metric info
		outfile = open(self.save_path / 'metric_info.json', 'w')
		json.dump(self.metric_info, outfile, indent=4)
		outfile.close()

	def _draw_trajectory_on_bev(self, bev_img, traj, target_point=None, dp_traj=None):
		"""
		Draw predicted trajectory on BEV image.
		
		BEV camera parameters:
		- Position: x=0, y=0, z=50 (50m height, looking down)
		- FOV: 50 degrees
		- Image size: 512x512
		
		Trajectory is in ego frame: [x, y] where x is forward, y is left (model convention)
		For BEV visualization, we negate y to convert to right-positive convention.
		BEV image: center is ego position, up is forward (negative x in image coords)
		
		Args:
			bev_img: numpy array (512, 512, 3) RGB image
			traj: numpy array (6, 2) trajectory points in ego frame [x_forward, y_left]
			target_point: numpy array (2,) target point in ego frame [x_forward, y_left], optional
			dp_traj: numpy array (6, 2) DP refined trajectory points in ego frame, optional
		
		Returns:
			bev_img: numpy array with trajectory drawn
		"""
		img_h, img_w = bev_img.shape[:2]  # 512, 512
		
		# BEV camera: z=50m, FOV=50 degrees
		# Calculate meters per pixel
		# FOV = 50 deg means the camera sees 50 degrees width/height
		# At z=50m, the ground coverage is: 2 * z * tan(FOV/2)
		fov_rad = np.deg2rad(50.0)
		ground_size = 2 * 50.0 * np.tan(fov_rad / 2)  # meters covered by the image
		meters_per_pixel = ground_size / img_w  # ~0.093 m/pixel
		
		# Image center is ego position
		cx, cy = img_w // 2, img_h // 2
		
		# Convert trajectory points to pixel coordinates
		# Model ego frame: x is forward, y is LEFT (positive y = left)
		# BEV image: center is ego, up (-row) is forward, right (+col) is right
		# Need to negate y to convert from left-positive to right-positive
		# So: pixel_col = cx + y / meters_per_pixel (negate y: left -> right, then right is +col)
		#     pixel_row = cy - x / meters_per_pixel (x forward -> -row, i.e., up)
		
		pixels = []
		for i in range(len(traj)):
			x, y = traj[i]  # x: forward, y: left (model convention)
			# Negate y for visualization: left-positive -> right-positive
			pixel_col = int(cx + y / meters_per_pixel)  # y_left negated: +y_left -> -col, so use + to flip
			pixel_row = int(cy - x / meters_per_pixel)
			pixels.append((pixel_col, pixel_row))
		
		# Draw trajectory using cv2
		# Draw lines connecting waypoints
		for i in range(len(pixels) - 1):
			pt1 = pixels[i]
			pt2 = pixels[i + 1]
			# Check if points are within image bounds
			if (0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and
				0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h):
				cv2.line(bev_img, pt1, pt2, (0, 255, 0), 2)  # Green line
		
		# Draw waypoints as circles
		for i, (col, row) in enumerate(pixels):
			if 0 <= col < img_w and 0 <= row < img_h:
				# Color gradient: start (red) -> end (blue)
				color_r = int(255 * (1 - i / (len(pixels) - 1)))
				color_b = int(255 * (i / (len(pixels) - 1)))
				cv2.circle(bev_img, (col, row), 5, (color_r, 0, color_b), -1)
		
		# Draw DP trajectory if provided (red color)
		if dp_traj is not None:
			dp_pixels = []
			for i in range(len(dp_traj)):
				x, y = dp_traj[i]  # x: forward, y: left (model convention)
				pixel_col = int(cx + y / meters_per_pixel)
				pixel_row = int(cy - x / meters_per_pixel)
				dp_pixels.append((pixel_col, pixel_row))
			
			# Draw DP trajectory lines (red)
			for i in range(len(dp_pixels) - 1):
				pt1 = dp_pixels[i]
				pt2 = dp_pixels[i + 1]
				if (0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and
					0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h):
					cv2.line(bev_img, pt1, pt2, (255, 0, 0), 2)  # Red line
			
			# Draw DP waypoints as circles (red with gradient to orange)
			for i, (col, row) in enumerate(dp_pixels):
				if 0 <= col < img_w and 0 <= row < img_h:
					# Color gradient: start (red) -> end (orange)
					color_g = int(128 * (i / (len(dp_pixels) - 1))) if len(dp_pixels) > 1 else 0
					cv2.circle(bev_img, (col, row), 4, (255, color_g, 0), -1)
		
		# Draw target point if provided (cyan/aqua color with larger circle)
		if target_point is not None:
			x, y = target_point[0], target_point[1]  # x: forward, y: left (model convention)
			# Negate y for visualization
			tp_col = int(cx + y / meters_per_pixel)  # Negate y: +y_left -> -col, use + to flip
			tp_row = int(cy - x / meters_per_pixel)
			if 0 <= tp_col < img_w and 0 <= tp_row < img_h:
				cv2.circle(bev_img, (tp_col, tp_row), 10, (0, 255, 255), -1)  # Cyan circle for target point
				cv2.circle(bev_img, (tp_col, tp_row), 12, (255, 255, 255), 2)  # White border
		
		# Draw ego position (center)
		cv2.circle(bev_img, (cx, cy), 8, (255, 255, 0), -1)  # Yellow circle for ego
		
		return bev_img

	def destroy(self):
		del self.net
		torch.cuda.empty_cache()

	def gps_to_location(self, gps):
		# gps content: numpy array: [lat, lon, alt]
		lat, lon = gps
		scale = math.cos(self.lat_ref * math.pi / 180.0)
		my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
		mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
		y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
		x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
		return np.array([x, y])


def lidar_to_ego_coordinate(lidar):
	"""
	Converts the LiDAR points given by the simulator into the ego agents
	coordinate system
	:param lidar: the LiDAR point cloud as provided in the input of run_step
	:return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
	coordinate system.
	"""
	yaw = np.deg2rad(-90.0)
	rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

	translation = np.array([0.0, 0.0, 2.5])

	# The double transpose is a trick to compute all the points together.
	ego_lidar = (rotation_matrix @ lidar[1][:, :3].T).T + translation

	return ego_lidar


def algin_lidar(lidar, translation, yaw):
	"""
	Translates and rotates a LiDAR into a new coordinate system.
	Rotation is inverse to translation and yaw
	:param lidar: numpy LiDAR point cloud (N,3)
	:param translation: translations in meters
	:param yaw: yaw angle in radians
	:return: numpy LiDAR point cloud in the new coordinate system.
	"""
	rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

	aligned_lidar = (rotation_matrix.T @ (lidar - translation).T).T

	return aligned_lidar


# ============== UKF Filter Functions ==============

def bicycle_model_forward(x, dt, steer, throttle, brake):
	"""
	Kinematic bicycle model.
	Numbers are the tuned parameters from World on Rails
	"""
	front_wb = -0.090769015
	rear_wb = 1.4178275

	steer_gain = 0.36848336
	brake_accel = -4.952399
	throt_accel = 0.5633837

	locs_0 = x[0]
	locs_1 = x[1]
	yaw = x[2]
	speed = x[3]

	if brake:
		accel = brake_accel
	else:
		accel = throt_accel * throttle

	wheel = steer_gain * steer

	beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
	next_locs_0 = locs_0.item() if hasattr(locs_0, 'item') else locs_0
	next_locs_0 = next_locs_0 + speed * math.cos(yaw + beta) * dt
	next_locs_1 = locs_1.item() if hasattr(locs_1, 'item') else locs_1
	next_locs_1 = next_locs_1 + speed * math.sin(yaw + beta) * dt
	next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
	next_speed = speed + accel * dt
	next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

	next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

	return next_state_x


def measurement_function_hx(vehicle_state):
	"""
	For now we use the same internal state as the measurement state
	:param vehicle_state: VehicleState vehicle state variable containing
		an internal state of the vehicle from the filter
	:return: np array: describes the vehicle state as numpy array.
		0: pos_x, 1: pos_y, 2: rotation, 3: speed
	"""
	return vehicle_state


def state_mean(state, wm):
	"""
	We use the arctan of the average of sin and cos of the angle to calculate
	the average of orientations.
	:param state: array of states to be averaged. First index is the timestep.
	:param wm: weights
	:return: averaged state
	"""
	x = np.zeros(4)
	sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
	sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
	x[0] = np.sum(np.dot(state[:, 0], wm))
	x[1] = np.sum(np.dot(state[:, 1], wm))
	x[2] = math.atan2(sum_sin, sum_cos)
	x[3] = np.sum(np.dot(state[:, 3], wm))

	return x


def measurement_mean(state, wm):
	"""
	We use the arctan of the average of sin and cos of the angle to
	calculate the average of orientations.
	:param state: array of states to be averaged. First index is the timestep.
	:param wm: weights
	:return: averaged measurement
	"""
	x = np.zeros(4)
	sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
	sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
	x[0] = np.sum(np.dot(state[:, 0], wm))
	x[1] = np.sum(np.dot(state[:, 1], wm))
	x[2] = math.atan2(sum_sin, sum_cos)
	x[3] = np.sum(np.dot(state[:, 3], wm))

	return x


def residual_state_x(a, b):
	y = a - b
	y[2] = t_u.normalize_angle(y[2])
	return y


def residual_measurement_h(a, b):
	y = a - b
	y[2] = t_u.normalize_angle(y[2])
	return y

