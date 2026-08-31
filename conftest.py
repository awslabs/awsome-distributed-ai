import pytest
import subprocess
import os


def pytest_addoption(parser):
    parser.addoption("--keep-artifacts", action="store")

@pytest.fixture
def change_test_dir(request):
    _orig_dir = os.getcwd()
    os.chdir(os.path.dirname(request.path))
    yield
    os.chdir(_orig_dir)

@pytest.fixture
def docker_build(change_test_dir, request):
    img_list = []
    def _build(name, dockerfile, test_tag=".test"):
        img_name=name + test_tag
        subprocess.check_call(['docker', 'build', '-t', img_name, '-f', dockerfile, '.'])
        img_list.append(img_name)
        return img_name

    yield _build
    if request.config.option.keep_artifacts is None:
        for img in img_list:
            subprocess.check_call(['docker', 'image', 'rm', img])

@pytest.fixture
def docker_run(change_test_dir):
    def _run(name, cmd, args=["--rm"]):
        subprocess.check_call(['docker', 'run'] + args + [name] + cmd)

    yield _run
