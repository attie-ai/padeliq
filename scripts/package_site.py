"""Build the public site and a Mac download from an explicit source allowlist."""
import argparse
import ast
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT_FILES = [
    'app.py', 'engine.py', 'geometry.py', 'requirements.txt',
    'README.md', 'TESTING.md', 'VERIFICATION.md', '.gitignore',
    '.streamlit/config.toml', 'tests/test_geometry.py', 'tests/test_pipeline.py',
]

def build(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in PROJECT_FILES:
        file = ROOT/name
        if not file.is_file():
            raise FileNotFoundError(name)
        if file.suffix == '.py':
            ast.parse(file.read_text(), filename=name)
    shutil.copyfile(ROOT/'index.html', destination/'index.html')
    (destination/'.nojekyll').touch()
    archive = destination/'PadelIQ-v0.1.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in PROJECT_FILES:
            z.write(ROOT/name, 'PadelIQ/'+name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError('Download archive verification failed.')
        assert set(z.namelist()) == {'PadelIQ/'+name for name in PROJECT_FILES}
    print(f'Built website and verified Mac download in {destination}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='dist')
    build(parser.parse_args().output)
