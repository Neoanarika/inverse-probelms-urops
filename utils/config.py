import yaml
import pickle
import hashlib 

def show(config):
    for item in config:
        if item != "exp_params":
            print(item)
            for key, val in config[item].items():
                print(f"    {key}: {val}")


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
      
      return config

def get_config_hash(config):
    return hashlib.sha256(pickle.dumps(config)).hexdigest()