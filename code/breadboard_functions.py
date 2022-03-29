def load_breadboard_client():
    import json 
    import sys
    import importlib.resources as pkg_resources
    from .. import configs as c 
    from .. import secrets as s
    with pkg_resources.path(c, "breadboard_path_config_local.json") as breadboard_config_path, pkg_resources.path(s, "breadboard_api_secret.json") as API_key_path:
        with open(breadboard_config_path) as breadboard_config_file:
            breadboard_config_dict = json.load(breadboard_config_file)
            breadboard_repo_path = breadboard_config_dict.get("breadboard_repo_path") 
            if(breadboard_repo_path is None):
                raise KeyError("The .json config does not contain variable breadboard_repo_path")
            sys.path.insert(0, breadboard_repo_path) 
            from breadboard import BreadboardClient
            bc = BreadboardClient(API_key_path)
    return bc