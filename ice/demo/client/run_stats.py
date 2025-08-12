import json
import os
import random


def main():
    result_file_path = os.getenv("ICE_RESULT_FILE")
    result = {
        "t1": {
            "age": (0, 17),
            "count": random.randint(100, 1000),
            "income": 0,
        },
        "t2": {
            "age": (18, 30),
            "count": random.randint(2000, 2500),
            "income": random.randint(10000, 70000),
        },
        "t3": {
            "age": (31, 60),
            "count": random.randint(3000, 10000),
            "income": random.randint(60000, 100000),
        },
    }
    with open(result_file_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"created result in {result_file_path}")


if __name__ == "__main__":
    main()
