import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("authorization_config_file", type=str)

    args = parser.parse_args()

    with open(args.authorization_config_file, "r") as f:
        config = json.load(f)

    # project_admin keeps all permissions (apart from byoc)
    config["permissions"]["project_admin"] = {
        "submit_job": "any",
        "clone_job": "any",
        "manage_job": "any",
        "download_job": "any",
        "view": "any",
        "operate": "any",
        "shell_commands": "any",
        "byoc": "none"
    }

    # disable byoc for everyone, including project_admin
    for role in config["permissions"]:
        config["permissions"][role]["byoc"] = "none"

    with open(args.authorization_config_file, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Disabled byoc for {args.authorization_config_file}")


if __name__ == "__main__":
    main()
