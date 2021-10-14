import yaml
import pickle
import hashlib 

def show(config):
    for item in config:
        if item != "exp_params":
            print(item)
            for key, val in config[item].items():
                print(f"    {key}: {val}")

def save(config, fname):
    with open(fname, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        
def get_config_base_model(fpath):
   with open(fpath, 'r') as file:
      try:
          config = yaml.safe_load(file)
      except yaml.YAMLError as exc:
          raise Exception(exc)
      
      # Check if config is valid
      assert "loss_params" in config
      assert "exp_params" in config
      assert "trainer_params" in config
      
      # Check if exp_params is valid
      assert "model_name" in config["exp_params"]
      assert "base_model" in config["exp_params"]
      assert "dataset" in config["exp_params"]
      assert "data_path" in config["exp_params"]
      assert "checkpoint_path" in config["exp_params"]
      assert "batch_size" in config["exp_params"]
      assert "latent_dim" in config["exp_params"]
      assert "image_shape" in config["exp_params"]

      # Check if trainer_params is valid
      assert "gpus" in config["trainer_params"]
      assert "max_epochs" in config["trainer_params"]
      
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
      assert "data_path" in config["exp_params"]
      assert "checkpoint_path" in config["exp_params"]
      assert "batch_size" in config["exp_params"]
      assert "image_shape" in config["exp_params"]

      # Check if base_model_params is valid
      assert "model_name" in config["base_model_params"]

      # Check if estimator_params is valid
      assert "estimator" in config["estimator_params"]

      # Check if operator_params is valid
      assert "operator" in config["operator_params"]
      
      return config

def get_config_hash(config):
    return hashlib.sha256(pickle.dumps(config)).hexdigest()