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
# Compile and reinstal llama-cpp-python with CUDA support
CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all-major" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
else
echo ERROR. \$PROCESSOR value must be one of 'cpu' or 'gpu'
exit 1;
fi
