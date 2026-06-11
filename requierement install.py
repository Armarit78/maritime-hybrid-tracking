import subprocess
import pkg_resources

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"  # Modifie si besoin

def install_torch_and_torchvision():
    subprocess.check_call([
        "pip", "install", "torch", "torchvision",
        "--index-url", TORCH_INDEX_URL
    ])

def get_installed_packages():
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def parse_requirements(requirements_file, exclude_list):
    requirements = []
    with open(requirements_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip().lower()
            if pkg_name not in exclude_list:
                requirements.append(line)
    return requirements

def install_missing_requirements(requirements):
    for req in requirements:
        try:
            subprocess.check_call(["pip", "install", req])
        except subprocess.CalledProcessError:
            print(f" ⚠️Erreur d'installation pour: {req}")

if __name__ == "__main__":
    print("Installation torch/torchvision (CUDA)...")
    install_torch_and_torchvision()

    print("Analyse des dépendances du requirements.txt...")
    exclude = ["torch", "torchvision"]
    installed = get_installed_packages()
    requirements = parse_requirements("requirements.txt", exclude)

    # Vérifie et installe seulement si le paquet manque ou la version ne correspond pas
    to_install = []
    for req in requirements:
        pkg_name = req.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip().lower()
        if pkg_name not in installed or (("==" in req) and installed.get(pkg_name) != req.split("==")[1]):
            to_install.append(req)

    if to_install:
        print("Installation des paquets manquants ou à mettre à jour:")
        install_missing_requirements(to_install)
    else:
        print("✅ Tous les paquets du requirements.txt (hors torch/torchvision) sont déjà installés")

    print("✅ Installation terminée.")

