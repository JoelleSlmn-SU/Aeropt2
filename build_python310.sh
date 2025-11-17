#!/bin/bash
set -e

# --------------------------- Configuration ---------------------------

PYTHON_VERSION="3.10.13"
OPENSSL_VERSION="1.1.1u"
PREFIX_BASE="$HOME/.opt"
TOOLS_DIR="$HOME/tools"
VENV_DIR="$HOME/.localpython"

PYTHON_SRC="$TOOLS_DIR/Python-${PYTHON_VERSION}"
OPENSSL_SRC="$TOOLS_DIR/openssl-${OPENSSL_VERSION}"
PYTHON_PREFIX="$PREFIX_BASE/python-${PYTHON_VERSION}"
OPENSSL_PREFIX="$PREFIX_BASE/openssl-${OPENSSL_VERSION}"

REQUIREMENTS_FILE="$HOME/aeropt/Scripts/requirements.txt"

# --------------------------- Download OpenSSL ---------------------------

mkdir -p "$TOOLS_DIR"
cd "$TOOLS_DIR"

echo "[INFO] Downloading OpenSSL $OPENSSL_VERSION..."
wget -nc https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz
tar -xzf openssl-${OPENSSL_VERSION}.tar.gz

cd "openssl-${OPENSSL_VERSION}"
echo "[INFO] Building OpenSSL..."

./config --prefix="$OPENSSL_PREFIX" --openssldir="$OPENSSL_PREFIX" shared zlib
make -j4
make install

# --------------------------- Download and Build Python ---------------------------

cd "$TOOLS_DIR"
wget -nc https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
tar -xzf Python-${PYTHON_VERSION}.tgz
cd "Python-${PYTHON_VERSION}"

echo "[INFO] Configuring Python with OpenSSL at $OPENSSL_PREFIX"

./configure --prefix="$PYTHON_PREFIX" \
  --with-openssl="$OPENSSL_PREFIX" \
  --with-ensurepip=install

make -j4
make install

# --------------------------- Create Virtual Env ---------------------------

"$PYTHON_PREFIX/bin/python3" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[INFO] Upgrading pip and installing requirements..."
pip install --upgrade pip setuptools wheel

if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "[WARNING] No requirements.txt found at $REQUIREMENTS_FILE"
fi

echo "[?] Python $PYTHON_VERSION installed at $PYTHON_PREFIX"
echo "[?] OpenSSL $OPENSSL_VERSION installed at $OPENSSL_PREFIX"
echo "[?] Virtual environment ready at $VENV_DIR"
