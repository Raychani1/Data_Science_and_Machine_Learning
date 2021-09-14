import os
import platform


def setup(requirements: object) -> None:
    current_platform = platform.system()
    if current_platform == 'Linux':
        os.system(
            f"pip install -r {requirements} | grep -v 'already satisfied'"
        )


if __name__ == '__main__':
    setup(os.path.join(os.getcwd(), 'requirements.txt'))
    print('\nInstalled Packages:')
    os.system("pip freeze")
    print()
