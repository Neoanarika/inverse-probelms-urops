import yaml
import pickle
import hashlib 

def get_config_base_model(fpath):
   with open(fpath, 'r') as file:
      try:
          config = yaml.safe_load(file)
      except yaml.YAMLError as exc:
          print(exc)
      
      # Check if config is valid
      assert "encoder_params" in config
      assert "decoder_params" in config 
      assert "loss_params" in config
      assert "exp_params" in config
      assert "trainer_params" in config

      return config

def get_config_ebm(fpath):
   with open(fpath, 'r') as file:
      try:
          config = yaml.safe_load(file)
      except yaml.YAMLError as exc:
          print(exc)
      
      # Check if config is valid
      assert "exp_params" in config
      assert "operator_params" in config 
      assert "estimator_params" in config
      assert "base_model_params" in config

      # Check exp params is valid 
      assert "dataset" in config["exp_params"]
      assert "base_model" in config["exp_params"]
      assert "operator" in config["exp_params"]
      assert "estimator" in config["exp_params"]

      assert type(config["exp_params"]["base_model"]) == str
      assert type(config["exp_params"]["dataset"]) == str
      assert type(config["exp_params"]["batch_size"]) == int

      return config

def get_config_hash(config):
    return hashlib.sha256(pickle.dumps(config)).hexdigest()