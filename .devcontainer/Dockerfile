# [Choice] Ubuntu version (use jammy or bionic on local arm64/Apple Silicon): jammy, focal, bionic
ARG VARIANT="jammy"
FROM nvcr.io/nvidia/pytorch:23.09-py3

# Options for setup script
ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="true"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
COPY .devcontainer/library-scripts/*.sh .devcontainer/library-scripts/*.env /tmp/library-scripts/
RUN yes | unminimize 2>&1 \
    && bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# Install dlib with cuda support (for face_recognition python package)
RUN git clone -b 'v19.21' --single-branch https://github.com/davisking/dlib.git dlib/ \
    && mkdir -p dlib/build \
    && cmake -Hdlib -Bdlib/build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 \
    && cmake --build dlib/build -- -j16 \
    && cd dlib \
    && python3 setup.py install --set BUILD_SHARED_LIBS=OFF
