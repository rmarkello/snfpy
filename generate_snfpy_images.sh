######################################################
# Generate a Dockerfile and Singularity recipe for building a snfpy container
# (https://snfpy.readthedocs.io/en/latest/).
# The Dockerfile and/or Singularity recipe installs most of snfpy's dependencies.
#
# Steps to build, upload, and deploy the snfpy docker and/or singularity image:
#
# 1. Create or update the Dockerfile and Singuarity recipe:
# bash generate_snfpy_images.sh
#
# 2. Build the docker image:
# docker build -t snfpy -f Dockerfile .
# OR
# bash generate_snfpy_images.sh docker
#
#    and/or singularity image:
# singularity build snfpy.simg Singularity
# OR
# bash generate_snfpy_images.sh singularity
#
#   and/or both:
# bash generate_snfpy_images.sh both
#
# 3. Push to Docker hub:
# (https://docs.docker.com/docker-cloud/builds/push-images/)
# export DOCKER_ID_USER="your_docker_id"
# docker login
# docker tag snfpy your_docker_id/snfpy:tag  # See: https://docs.docker.com/engine/reference/commandline/tag/
# docker push your_docker_id/snfpy:tag
#
# 4. Pull from Docker hub (or use the original):
# docker pull your_docker_id/snfpy
#
# In the following, the Docker container can be the original (snfpy)
# or the pulled version (your_docker_id/snfpy:tag), and is given access to /Users/snfpy
# on the host machine.
#
# 5. Enter the bash shell of the Docker container, and add port mappings:
# docker run --rm -ti -v /Users/snfpy:/home/snfpy -p 8888:8888 -p 5000:5000 your_docker_id/snfpy
#
#
###############################################################################

image="kaczmarj/neurodocker:0.6.0"

set -e

generate_docker() {
 docker run --rm ${image} generate docker \
            --base neurodebian:stretch-non-free \
            --pkg-manager apt \
            --install git \
            --user=snfpy \
            --miniconda \
               conda_install="python=3.7 notebook ipython" \
               pip_install='git+https://github.com/rmarkello/snfpy.git@master ipywidgets ipyevents jupytext seaborn pandas' \
               create_env='snfpy' \
               activate=true \
            --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
            --entrypoint="/neurodocker/startup.sh"
}

generate_singularity() {
  docker run --rm ${image} generate singularity \
            --base neurodebian:stretch-non-free \
            --pkg-manager apt \
            --install git \
            --user=snfpy \
            --miniconda \
               conda_install="python=3.7 notebook ipython" \
               pip_install='git+https://github.com/rmarkello/snfpy.git@master ipywidgets ipyevents jupytext seaborn pandas' \
               create_env='snfpy' \
               activate=true \
            --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
            --entrypoint="/neurodocker/startup.sh"
 }

# generate files
generate_docker > Dockerfile
generate_singularity > Singularity

# check if images should be build locally or not
if [ '$1' = 'docker' ]; then
 echo "docker image will be build locally"
 # build image using the saved files
 docker build -t snfpy .
elif [ '$1' = 'singularity' ]; then
 echo "singularity image will be build locally"
 # build image using the saved files
 singularity build snfpy.simg Singularity
elif [ '$1' = 'both' ]; then
 echo "docker and singularity images will be build locally"
 # build images using the saved files
 docker build -t snfpy .
 singularity build snfpy.simg Singularity
else
echo "Image(s) won't be build locally."
fi
