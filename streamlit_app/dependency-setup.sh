set -euxo pipefail



if [[ -z "$PROCESSOR" ]] ; then 
echo PROCESSOR not provided. Falling back to cpu... Set PROCESSOR = "gpu" and rebuild to install CUDA dependencies.
else 
echo PROCESSOR is $PROCESSOR ;
fi

if [ "$PROCESSOR" = "cpu" ] ; then
echo PROCESSOR = 'cpu'. Skipping CUDA dependencies... Set PROCESSOR = "gpu" and rebuild to install CUDA dependencies.
exit 0;
elif [ "$PROCESSOR" = "gpu" ] ; then
# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb && \
dpkg -i cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb && \
cp /var/cuda-repo-debian12-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
add-apt-repository contrib && \
apt-get update && \
apt-get -y install cuda-toolkit-12-3 && \
rm -rf /var/lib/apt/lists/*

# Compile and reinstal llama-cpp-python with CUDA support
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
pip install -r requirements-cuda.txt ;
else
echo ERROR. \$PROCESSOR value must be one of 'cpu' or 'gpu'
exit 1;
fi
