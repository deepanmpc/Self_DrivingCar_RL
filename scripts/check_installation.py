"""Check if all required packages are installed."""

import sys


def check_python_version():
    """Check Python version meets minimum requirements."""
    version = sys.version_info
    minimum = (3, 8)
    if version < minimum:
        print(f"Python {version.major}.{version.minor}.{version.micro} < required {minimum[0]}.{minimum[1]}")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro} ✓")
    return True


def check_packages():
    """Check if required packages are importable."""
    packages = [
        ("torch", "PyTorch"),
        ("gymnasium", "Gymnasium"),
        ("numpy", "NumPy"),
        ("shapely", "Shapely"),
    ]
    
    all_ok = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"{name} ✓")
        except ImportError:
            print(f"{name} ✗ MISSING")
            all_ok = False
    
    return all_ok


def main():
    """Run all checks and report results."""
    print("=== Installation Check ===")
    python_ok = check_python_version()
    packages_ok = check_packages()
    
    if python_ok and packages_ok:
        print("\nAll checks passed! ✓")
        return 0
    else:
        print("\nSome checks failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())