import yaml

def get_config(fpath):
   with open(fpath, 'r') as file:
      try:
          config = yaml.safe_load(file)
      except yaml.YAMLError as exc:
          print(exc)
      return config