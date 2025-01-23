import subprocess
import sys
import os
import platform

def check_system_compatibility():
    """Check system and Python compatibility."""
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")

def install_package(pip_path, package):
    """Install a single package and handle errors."""
    try:
        subprocess.run([pip_path, "install", package], check=True)
        print(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to install {package}: {e}")
        return False

def create_virtual_environment():
    """Create a virtual environment."""
    venv_name = "sleep_fed_env"
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        print(f"Virtual environment '{venv_name}' created successfully")
        return venv_name
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

def install_requirements(venv_name):
    """Install required packages individually for better error handling."""
    pip_path = os.path.join(venv_name, "Scripts", "pip.exe") if sys.platform == "win32" else os.path.join(venv_name, "bin", "pip")
    
    # Upgrade pip first
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    
    with open("requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    successful_installations = []
    failed_installations = []
    
    for req in requirements:
        if install_package(pip_path, req):
            successful_installations.append(req)
        else:
            failed_installations.append(req)
    
    print("\nInstallation Summary:")
    print(f"Successfully installed {len(successful_installations)} packages")
    if failed_installations:
        print(f"Failed to install {len(failed_installations)} packages:")
        for pkg in failed_installations:
            print(f"  - {pkg}")

def setup_project_structure():
    """Create project directory structure."""
    directories = [
        "data/raw",
        "data/processed",
        "data/interim",
        "models",
        "src/data_processing",
        "src/models",
        "src/federated",
        "src/visualization",
        "notebooks",
        "tests",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create __init__.py files in src directories
    src_dirs = [d for d in directories if d.startswith('src/')]
    for src_dir in src_dirs:
        init_file = os.path.join(src_dir, '__init__.py')
        if not os.path.exists(init_file):
            open(init_file, 'a').close()
            print(f"Created __init__.py in {src_dir}")

def main():
    """Main setup function."""
    print("Starting development environment setup...")
    
    # Check system compatibility
    check_system_compatibility()
    
    # Create virtual environment
    venv_name = create_virtual_environment()
    
    # Install requirements
    install_requirements(venv_name)
    
    # Setup project structure
    setup_project_structure()
    
    print("\nSetup completed!")
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print(f"    {venv_name}\\Scripts\\activate")
    else:
        print(f"    source {venv_name}/bin/activate")
    
    print("\nNext steps:")
    print("1. Activate the virtual environment")
    print("2. Start with data preprocessing by running the scripts in src/data_processing")
    print("3. Check the notebooks directory for example usage")

if __name__ == "__main__":
    main()